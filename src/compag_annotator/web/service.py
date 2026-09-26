"""Application workflows joining independent providers, data, jobs and formats."""
from __future__ import annotations
import copy,json,shutil,zipfile
from pathlib import Path
from compag_annotator import __version__
from compag_annotator.core.catalog import Catalog
from compag_annotator.core.projects import ConflictError
from compag_annotator.jobs.runner import JobRunner,AwaitingReview
from compag_annotator.boundary.preparation import BoundaryPreparation
from compag_annotator.storage.files import atomic,digest,uid,now,safe_extract,safe_child
from compag_annotator.training.snapshots import readiness,build_snapshot

class Service:
    def __init__(self,data_dir,qa_mode=False):
        self.data_dir=Path(data_dir);self.catalog=Catalog(data_dir);self.jobs=JobRunner(data_dir);self.boundary=BoundaryPreparation();self.qa_mode=qa_mode;self.actor='automated_qa' if qa_mode else 'human';self._manager=None
    @property
    def manager(self):
        if self._manager is None:
            from compag_annotator.models.manager import ModelManager
            self._manager=ModelManager(self.data_dir)
        return self._manager
    def project(self,pid):return self.catalog.get(pid)
    def submit(self,kind,payload,key=None):
        payload=copy.deepcopy(payload)
        with self.jobs.lock:
            if payload.get('project_id'):self.project(payload['project_id'])
            return self.jobs.submit(kind,payload,lambda directory,progress,cancel:self.work(kind,payload,directory,progress,cancel),key)
    def submit_generation(self,project_id,body,key=None):
        from compag_annotator.core.generation import freeze_generation,request_identity
        key=request_identity(project_id,body,key)
        with self.jobs.lock:
            previous=next((j for j in self.jobs.list() if j.get('idempotency_key')==key),None)
            if previous:return previous
            payload=freeze_generation(self,project_id,body)
            return self.submit('generate',payload,key)
    def submit_annotation_boundary(self,project_id,body,key=None):
        if body.get('explicit_opt_in') is not True:raise ValueError('Explicitly opt in to boundary checking for this selected scope')
        project=self.project(project_id);state=project.state();image=project.image(state,body['image_id'])
        if body.get('expected_revision')!=state['revision']:raise ConflictError('Project changed before boundary scope was planned')
        rows=project.annotations(image['id'])
        if body.get('layer_id'):
            if body.get('annotation_ids'):raise ValueError('Select a layer or individual objects, not both')
            layer=next((l for l in state.get('generation_layers',[]) if l['id']==body['layer_id'] and l['image_id']==image['id']),None)
            if not layer:raise ValueError('This generation layer does not belong to the selected image')
            selected=[r for r in rows if r.get('layer_id')==layer['id'] and r['status'] not in ('rejected','superseded')]
        else:
            ids=body.get('annotation_ids')
            if not isinstance(ids,list) or not ids or not all(isinstance(i,str) for i in ids) or len(ids)!=len(set(ids)):raise ValueError('Select distinct generated proposals')
            selected=[r for r in rows if r['id'] in ids and r['status'] not in ('rejected','superseded')]
            if len(selected)!=len(ids):raise ValueError('Boundary scope contains a missing, inactive or cross-image object')
            known_layers={l['id']:l for l in state.get('generation_layers',[]) if l['image_id']==image['id']}
            if any(r.get('layer_id') not in known_layers or r['source'].get('generation_id')!=known_layers[r['layer_id']]['generation_id'] for r in selected):
                raise ValueError('Annotation-time boundary checking requires server-created automatic-generation membership')
        if not selected:raise ValueError('No live generated proposals in the selected scope')
        limit=body.get('limit',3)
        if type(limit) is not int or not 1<=limit<=20:raise ValueError('Boundary scope limit must be 1–20 candidates')
        model=next((m for m in self.manager.models() if m['id']==body.get('model_id') and m['provider'] in ('sam2','sam3')),None)
        if model is None:raise ValueError('Choose an installed SAM2/SAM3 model')
        payload={'project_id':project_id,'image_id':image['id'],'model_id':model['id'],'annotation_ids':[r['id'] for r in selected],
                 'annotation_revisions':{r['id']:r['revision'] for r in selected},'limit':limit,'device':body.get('device','cpu'),
                 'provider_settings':body.get('provider_settings',{}),'explicit_opt_in':True}
        return self.submit('boundary_annotation',payload,key)
    def generation_boundary(self,project,body,directory,progress,cancel):
        layer_ids={l['id'] for l in body['layers']}
        ids=[r['id'] for r in project.annotations() if r.get('layer_id') in layer_ids and r['status'] not in ('rejected','superseded')]
        grant=self.boundary.authorize(project,job_kind='generate',job_id=directory.name,annotation_ids=ids,explicit_opt_in=True,limit=body['boundary_limit'])
        settings={'phase':'automatic_mask_annotation','boundary_opt_in':True,'boundary_model_id':body['model_id'],'device':body['device'],
                  'provider_settings':{'precision':body['settings'].get('precision','float32')}}
        try:return self.boundary.prepare(project,self.manager,settings,progress,cancel,authorization=grant)
        except AwaitingReview as error:return {**error.result,'awaiting_review':True}
    def activate(self,project,body):
        if body.get('confirm') is not True:raise ValueError('Confirm use of this model for future inference')
        model=next((m for m in self.manager.models() if m['id']==body['model_id']),None)
        if not model or model.get('provider')!='yolo':raise ValueError('Choose a registered YOLO segmentation model')
        mapping=body.get('mapping') or model.get('class_mapping') or {}
        if not isinstance(mapping,dict):raise ValueError('Explicit model-index to project-class mapping is required')
        known_names=model.get('class_names')
        if known_names:
            known_indices={str(k) for k in known_names} if isinstance(known_names,dict) else {str(i) for i in range(len(known_names))}
            if not set(map(str,mapping)).issubset(known_indices):raise ValueError('Mapping references an index absent from this model')
        with project.edit(body.get('expected_revision'),'model_activate',self.actor) as (db,state):
            valid={c['id'] for c in state['classes'] if not c['archived']}
            if any(not str(k).isdigit() or v not in valid for k,v in mapping.items()):raise ValueError('Mapping contains an invalid model index or project class')
            state['model_history'].append({'model_id':state['active_model_id'],'mapping':state['active_model_mapping'],'class_version':state['active_model_class_version'],'changed_at':now()})
            state.update(active_model_id=model['id'],active_model_mapping=mapping,active_model_class_version=state['class_version'])
            if not any(m['id']==model['id'] for m in state['models']):state['models'].append(model)
        return project.public()
    def rollback(self,project,body):
        if body.get('confirm') is not True:raise ValueError('Confirm model rollback')
        with project.edit(body.get('expected_revision'),'model_rollback',self.actor) as (db,state):
            if not state['model_history']:raise ValueError('No earlier model activation')
            previous=state['model_history'].pop();state.update(active_model_id=previous['model_id'],active_model_mapping=previous['mapping'],active_model_class_version=previous['class_version'])
        return project.public()
    def work(self,kind,body,directory,progress,cancel):
        project=self.project(body['project_id']) if body.get('project_id') else None
        if kind=='generate':
            from compag_annotator.core.generation import run_generation
            return run_generation(self,body,directory,progress,cancel)
        if kind=='boundary_annotation':
            current={r['id']:r['revision'] for r in project.annotations(body['image_id']) if r['id'] in body['annotation_ids']}
            if current!=body['annotation_revisions']:raise ConflictError('Boundary selection changed; prepare a new explicit scope')
            grant=self.boundary.authorize(project,job_kind=kind,job_id=directory.name,annotation_ids=body['annotation_ids'],explicit_opt_in=body['explicit_opt_in'],limit=body['limit'])
            return self.boundary.prepare(project,self.manager,{'phase':'automatic_mask_annotation','boundary_opt_in':True,'boundary_model_id':body['model_id'],
                'device':body['device'],'provider_settings':body['provider_settings']},progress,cancel,authorization=grant)
        if kind=='import_images':
            result=project.add_images(body['paths'],body.get('storage','copy'),progress,cancel,body.get('display_names'))
            if body.get('uploaded'):
                for value in body['paths']:
                    p=Path(value).resolve()
                    if p.is_relative_to((self.data_dir/'uploads').resolve()) and p.is_file():
                        shutil.rmtree(p.parent)
                        try:p.parent.parent.rmdir()
                        except OSError:pass
            return result
        if kind=='install_model':return self.manager.install(body['provider'],body.get('architecture'),consent=body.get('consent') is True,download_weights=body.get('download_weights',False),progress=progress,cancel=cancel)
        if kind in ('assist','test_model'):
            if kind == 'assist' and body.get('mode', 'single') not in ('single', 'independent'):
                raise ValueError('Choose a supported SAM prompt mode')
            if kind == 'assist' and body.get('mode') == 'independent':
                from compag_annotator.core.assist import independent_preview
                return independent_preview(self,project,body,progress,cancel)
            state=project.state();image=project.image(state,body['image_id'])
            expected=body.get('expected_revision',state['revision'])
            if expected!=state['revision']:raise ConflictError('Image/project changed before prompting')
            result=self.manager.infer(body['model_id'],str(safe_child(project.path,image['path'])),points=body.get('points'),labels=body.get('labels'),box=body.get('box'),device=body.get('device','cpu'),settings=body.get('settings'),progress=progress,cancel=cancel)
            if project.state()['revision']!=expected:raise ConflictError('Discarded stale provider response after project changed')
            result['binding']={'image_id':image['id'],'image_sha256':image['sha256'],'project_revision':expected,'annotation_id':body.get('annotation_id'),'model_id':body['model_id'],'class_id':body.get('class_id'),'points':body.get('points'),'labels':body.get('labels'),'box':body.get('box')}
            result['preview_only']=True;return result
        if kind=='predict':
            from compag_annotator.geometry import deduplicate
            state=project.state();model_id=body.get('model_id') or state['active_model_id']
            if not model_id:raise ValueError('Select and activate a segmentation model first')
            if model_id==state['active_model_id'] and state['active_model_class_version']!=state['class_version']:raise ValueError('Class schema changed; review model-to-class mapping before inference')
            expected=body.get('expected_revision',state['revision'])
            if state['revision']!=expected:raise ConflictError('Project changed before inference')
            if body.get('boundary_opt_in') is True:
                if not any(m['id']==body.get('boundary_model_id') and m['provider'] in ('sam2','sam3') for m in self.manager.models()):
                    raise ValueError('Choose an installed SAM model for the optional Boundary check')
                if type(body.get('boundary_limit',3)) is not int or not 1<=body.get('boundary_limit',3)<=20:
                    raise ValueError('Boundary limit must be 1–20 candidates')
            outputs=[];generated_ids=[]
            for index,iid in enumerate(body['image_ids']):
                if cancel():raise InterruptedError('Prediction cancelled')
                image=project.image(state,iid)
                result=self.manager.infer(model_id,str(safe_child(project.path,image['path'])),device=body.get('device','cpu'),settings=body.get('settings'),progress=progress,cancel=cancel)
                candidates=[]
                for row in result.get('annotations',[]):
                    row=copy.deepcopy(row);index_key=str(row.get('model_class_index',row.get('class_index','')))
                    row['class_id']=state['active_model_mapping'].get(index_key) if model_id==state['active_model_id'] else None
                    row.setdefault('source',{});row['source'].update(kind='yolo',model_id=model_id,model_class_index=row.get('model_class_index',row.get('class_index')),model_sha256=result.get('loaded_checkpoint_sha256',result.get('model_sha256')))
                    candidates.append(row)
                if not result.get('tile_plan') and 'tiling' not in (body.get('settings') or {}):
                    candidates=deduplicate(candidates,image['width'],image['height'])
                outputs.append({'image_id':iid,'predictions':candidates,'receipt':{k:v for k,v in result.items() if k!='annotations'}})
                progress({'stage':'full_image_inference','completed':index+1,'total':len(body['image_ids'])})
            # One revision-checked commit; no accepted object is overwritten.
            with project.edit(expected,'model_proposals','provider') as (db,current):
                for output in outputs:
                    image=project.image(current,output['image_id'])
                    for proposal in output['predictions']:
                        row={'id':uid(),'image_id':image['id'],'class_id':proposal['class_id'],'geometry':project._geometry(proposal['geometry'],image),'status':'proposal','source':proposal['source'],'revision':current['revision']+1,'review_actor':None,'human_verified':False}
                        if 'score' in proposal:row['score']=proposal['score']
                        project._put(db,row)
                        generated_ids.append(row['id'])
                    image.update(complete=False,review_actor=None,revision=current['revision']+1)
            atomic(directory/'inference_receipt.json',outputs)
            response={'images':[{'image_id':o['image_id'],'proposals':len(o['predictions']),**o['receipt']} for o in outputs],'boundary_invocations':0}
            if body.get('boundary_opt_in') is True:
                grant=self.boundary.authorize(project,job_kind='predict',job_id=directory.name,annotation_ids=generated_ids,explicit_opt_in=True,limit=body.get('boundary_limit',3))
                response['boundary_invocations']=1
                try:
                    response['boundary']=self.boundary.prepare(project,self.manager,{**body,'phase':'prediction_boundary_check'},progress,cancel,authorization=grant)
                except Exception as error:
                    trace=getattr(error,'result',{})
                    atomic(directory/'boundary_trace.json',trace)
                    error.result={**response,'boundary':trace,'proposals_saved':True}
                    raise
                atomic(directory/'boundary_trace.json',response['boundary'])
            return response
        if kind=='train':
            if body.get('qa_smoke') and not self.qa_mode:raise ValueError('Automated QA datasets are permitted only in explicit QA sessions')
            for key,low,high in [('epochs',1,10000),('imgsz',32,8192),('batch',1,1024)]:
                value=body.get(key,{'epochs':3,'imgsz':640,'batch':2}[key])
                if type(value) is not int or not low<=value<=high:raise ValueError('Invalid '+key)
                body[key]=value
            state=project.state()
            active_ids={i['id'] for i in state['images'] if not i.get('removed')}
            if any(p['status']=='pending' and p['image_id'] in active_ids for p in state['boundary_proposals']):raise AwaitingReview('Review pending boundary suggestions before freezing training data')
            boundary={'invoked':False,'allowed_phase':'training_preparation','explicit_job_opt_in':False}
            if body.get('boundary_opt_in') is True:
                ids=[row['id'] for row in project.document(reviewed_only=True)['annotations']]
                grant=self.boundary.authorize(project,job_kind='train',job_id=directory.name,annotation_ids=ids,explicit_opt_in=True,limit=body.get('boundary_limit',3))
                boundary=self.boundary.prepare(project,self.manager,{**body,'phase':'training_preparation'},progress,cancel,authorization=grant)
            if body.get('resume_job_id'):
                old=self.jobs.get(body['resume_job_id'])
                if old['kind']!='train' or old['payload'].get('project_id')!=body['project_id']:raise ValueError('Resume checkpoint belongs to another project/job')
                original_dataset=self.jobs.root/old['id']/'dataset'
                receipt=json.loads((original_dataset/'snapshot.json').read_text())
                for name,h in receipt['file_hashes'].items():
                    if digest(safe_child(original_dataset,name))!=h:raise ValueError('Resume snapshot files changed')
                choices=list((self.jobs.root/old['id']/'training').rglob('last.pt'))
                if len(choices)!=1:raise ValueError('No unambiguous saved last checkpoint is available for resume')
                resumed=self.manager.register('yolo',choices[0],architecture='custom-seg',trust=True,class_mapping=receipt['class_mapping'])
                snapshot={'dataset_yaml':str(original_dataset/'data.yaml'),'receipt':receipt,'snapshot_sha256':digest(original_dataset/'snapshot.json')}
                settings={**receipt['settings'],'resume_checkpoint':str(choices[0]),'resume':True}
                body['model_id']=resumed['id']
                atomic(directory/'resume_binding.json',{'original_job_id':old['id'],'snapshot_sha256':snapshot['snapshot_sha256'],'checkpoint_sha256':digest(choices[0]),'new_edits_excluded':True})
            else:
                snapshot=build_snapshot(project,directory/'dataset',body)
                settings=dict(body)
            atomic(directory/'boundary_trace.json',boundary)
            settings.update(class_mapping=snapshot['receipt']['class_mapping'],class_version=snapshot['receipt']['class_version'],snapshot_sha256=snapshot['snapshot_sha256'])
            if snapshot['receipt'].get('training_layout',{}).get('mode')=='tiles_512':
                settings['training_layout']=snapshot['receipt']['training_layout']
            provider_keys={'epochs','imgsz','batch','device','seed','workers','patience','lr0','optimizer','rect','cache','close_mosaic','mosaic','mixup','cutmix','copy_paste','scale','translate','fliplr','flipud','degrees','shear','perspective','deterministic','amp','class_mapping','qa_smoke','round_id','dataset_id','worker_timeout','full_instance','resume','resume_checkpoint','precision','reserve_free_bytes','class_version','snapshot_sha256'}
            provider_settings={k:v for k,v in settings.items() if k in provider_keys}
            if 'training_layout' in settings:provider_settings['training_layout']=settings['training_layout']
            result=self.manager.train(body['model_id'],snapshot['dataset_yaml'],directory/'training',provider_settings,progress=progress,cancel=cancel)
            model_id=result.get('model_id') or result.get('model',{}).get('id')
            if not model_id:raise ValueError('Training did not register a validated model')
            result.update(snapshot_sha256=snapshot['snapshot_sha256'],dataset_snapshot=str(Path(snapshot['dataset_yaml']).parent/'snapshot.json'),boundary=boundary,qa_smoke=body.get('qa_smoke',False),human_uat=False if body.get('qa_smoke') else None)
            with project.edit(None,'training_complete','provider') as (db,state):
                model=next((m for m in self.manager.models() if m['id']==model_id),None)
                if model and not any(m['id']==model_id for m in state['models']):state['models'].append(model)
                if body.get('round_id'):
                    r=next(r for r in state['rounds'] if r['id']==body['round_id']);r.update(training_status='complete',trained_model_id=model_id,dataset_snapshot_sha256=snapshot['snapshot_sha256'])
            if body.get('auto_activate') is True:
                self.activate(project,{'model_id':model_id,'mapping':snapshot['receipt']['class_mapping'],'confirm':True,'expected_revision':project.state()['revision']})
                result['activated']=True
            else:result['activated']=False
            return result
        if kind=='export':
            fmt=body['format']
            if fmt=='native':return self.catalog.backup(project,directory/'project.compag.zip',body.get('include_images',True))
            from compag_annotator.formats import export_annotations
            if 'annotation_scope' in body:
                from compag_annotator.core.dataset_export import plan_dataset,write_dataset_help
                if type(body.get('expected_revision')) is not int:
                    raise ValueError('Check the export before creating a ZIP')
                progress({'stage':'Checking annotation selection'})
                summary,document=plan_dataset(project,body,geometry=True)
                if not summary['ready']:raise ValueError(' '.join(summary['errors']))
                out=directory/'dataset'
                progress({'stage':'Writing annotations and image copies' if body.get('include_images') else 'Writing annotation files'})
                result=export_annotations(fmt,document,out,include_images=body.get('include_images',False),allow_lossy=body.get('allow_lossy',False),png_variant=body.get('png_variant','per_instance'),conflict_policy=body.get('conflict_policy','error'))
                write_dataset_help(out,document,summary,result)
                archive=directory/(fmt+'-dataset.zip');files=sorted(p for p in out.rglob('*') if p.is_file())
                with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                    for index,path in enumerate(files):
                        if cancel():raise RuntimeError('Export cancelled')
                        z.write(path,path.relative_to(out))
                        progress({'stage':'Packing dataset ZIP','completed':index+1,'total':len(files)})
                return {'artifact':str(archive),'sha256':digest(archive),'format':fmt,'report':result,'summary':summary,
                        'archive_bytes':archive.stat().st_size,'uncompressed_bytes':sum(p.stat().st_size for p in files),
                        'images_included':summary['images_included'],'weights_included':False,'history_included':False}
            state=project.state();scope=body.get('scope','all');ids=None
            if scope=='selected':ids=body.get('image_ids',[])
            elif scope=='round':ids=next((r['image_ids'] for r in state['rounds'] if r['id']==body.get('round_id')),[])
            elif scope=='cumulative':ids=[i['id'] for i in state['images'] if i['role'] in ('pool','train') and i['complete'] and not i.get('training_excluded')]
            elif scope!='all':raise ValueError('Invalid export scope')
            document=project.document(ids,body.get('reviewed_only',True));out=directory/'export'
            if body.get('reviewed_only',True) and not document['images']:
                raise ValueError('No fully reviewed images in this export scope. Accept or reject the remaining proposals and explicitly mark the full image reviewed, or deliberately choose a draft-inclusive export.')
            result=export_annotations(fmt,document,out,include_images=body.get('include_images',True),allow_lossy=body.get('allow_lossy',False),png_variant=body.get('png_variant','per_instance'),conflict_policy=body.get('conflict_policy','error'))
            atomic(out/'compag_export_scope.json',{'scope':scope,'draft_inclusive':document['draft_inclusive'],'project_revision':document['revision'],'images':len(document['images']),'annotations':len(document['annotations']),'boundary_recovery_invoked':False})
            archive=directory/(fmt+'.zip')
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                for p in out.rglob('*'):
                    if p.is_file():z.write(p,p.relative_to(out))
            return {'artifact':str(archive),'sha256':digest(archive),'report':result}
        if kind=='import_annotations':
            from compag_annotator.formats import import_annotations
            path=Path(body['path']);input_path=path
            if zipfile.is_zipfile(path):input_path=safe_extract(path,directory/'input')
            doc=project.document(reviewed_only=False);result=import_annotations(body['format'],input_path,doc['images'],doc['classes'],options=body.get('options'))
            with project.edit(None,'annotation_import','import') as (db,state):
                classes=result.get('classes',[])
                known={c['id'] for c in state['classes']}
                before_class_count=len(state['classes'])
                for c in classes:
                    if c['id'] not in known:state['classes'].append({**c,'archived':False,'order':len(state['classes']),'color':c.get('color','#36c8aa'),'shortcut':c.get('shortcut','')});known.add(c['id'])
                if len(state['classes'])!=before_class_count:state['class_version']+=1
                for annotation in result.get('annotations',[]):
                    image=project.image(state,annotation['image_id']);row={**annotation,'id':uid(),'geometry':project._geometry(annotation['geometry'],image),'status':'draft','review_actor':None,'human_verified':False,'revision':state['revision']+1,'source':{'kind':'import','format':body['format'],'original_id':annotation.get('id')}}
                    project._put(db,row);image.update(complete=False,review_actor=None)
            return {'annotations':len(result.get('annotations',[])),'loss_report':result.get('loss_report'), 'review_status':'unreviewed'}
        if kind=='restore':return self.catalog.restore(body['path']).public()
        if kind=='diagnostics':
            diagnostic={'version':__version__,'projects':len(self.catalog.records()),'jobs':[{'kind':j['kind'],'status':j['status']} for j in self.jobs.list()],'boundary_invocations':self.boundary.invocations,'paths_and_tokens_omitted':True,'qa_mode':self.qa_mode}
            atomic(directory/'diagnostics.json',diagnostic);return {'artifact':str(directory/'diagnostics.json'),'preview':diagnostic}
        raise ValueError('Unknown job kind')
