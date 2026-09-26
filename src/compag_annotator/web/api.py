"""Local UI API. Mutations are protected by the app capability middleware."""
import json,os,platform,shutil,sys
from pathlib import Path
from fastapi import APIRouter,Request,UploadFile,File,Form
from fastapi.responses import FileResponse
from compag_annotator import __version__
from compag_annotator.config import MAX_UPLOAD_BYTES
from compag_annotator.storage.files import uid,safe_child,digest,atomic
from compag_annotator.core.projects import ConflictError
from compag_annotator.training.snapshots import readiness


def router(service,token):
    api=APIRouter(prefix='/api')
    def project(pid):return service.project(pid)
    def submit(kind,body,request=None):return service.submit(kind,body,request.headers.get('idempotency-key') if request else None)
    def expected(body):
        if type(body.get('expected_revision')) is not int:raise ValueError('Expected project revision is required')
    async def save_uploads(files):
        folder=service.data_dir/'uploads'/uid();folder.mkdir(parents=True);paths=[];total=0
        try:
            for uploaded in files:
                # Preserve basename for display, disambiguate equal names in separate directories.
                name=Path((uploaded.filename or 'upload').replace('\\','/')).name
                if name in ('','.','..'):raise ValueError('Invalid upload name')
                stored_name=name
                while len(stored_name.encode('utf-8'))>230:stored_name=stored_name[:-1]
                target=folder/uid()/stored_name;target.parent.mkdir()
                with target.open('xb') as out:
                    while part:=await uploaded.read(1024**2):
                        total+=len(part)
                        if total>MAX_UPLOAD_BYTES:raise ValueError('Upload exceeds 256 MiB limit')
                        out.write(part)
                atomic(target.parent/'upload_name.json',{'name':name})
                paths.append(str(target))
            return paths
        except BaseException:shutil.rmtree(folder);raise
    @api.get('/session')
    def session():return {'token':token,'version':__version__,'actor':service.actor,'qa_mode':service.qa_mode}
    @api.get('/projects')
    def projects():return service.catalog.records()
    @api.post('/projects')
    def create_project(body:dict):return service.catalog.create(body.get('name','Untitled project'),body.get('classes',[]),body.get('path')).public()
    @api.post('/projects/open')
    def open_project(body:dict):return service.catalog.open(body['path']).public()
    @api.post('/projects/restore')
    async def restore_project(archive:UploadFile=File(...)):
        paths=await save_uploads([archive]);return submit('restore',{'path':paths[0]})
    @api.get('/projects/{pid}')
    def get_project(pid:str):return project(pid).public()
    @api.get('/projects/{pid}/deletion-preview')
    def project_deletion_preview(pid:str):
        from compag_annotator.core.deletion import preview_project_deletion
        return preview_project_deletion(service,pid)
    @api.delete('/projects/{pid}')
    def delete_project(pid:str,body:dict):
        from compag_annotator.core.deletion import delete_project
        expected(body);return delete_project(service,pid,body)
    @api.patch('/projects/{pid}')
    def rename_project(pid:str,body:dict):
        from compag_annotator.core.projects import clean_name
        expected(body);p=project(pid)
        with p.edit(body['expected_revision'],'project_rename',service.actor) as (db,state):state['name']=clean_name(body['name'])
        service.catalog.register(p);return p.public()
    @api.post('/projects/{pid}/classes/preview')
    def class_preview(pid:str,body:dict):return project(pid).class_change(body,service.actor,preview=True)
    @api.post('/projects/{pid}/classes')
    def class_change(pid:str,body:dict):expected(body);return project(pid).class_change(body,service.actor)
    @api.post('/projects/{pid}/processing/preview')
    def processing_preview(pid:str,body:dict):
        from compag_annotator.geometry import plan_tiles
        from compag_annotator.config import DEFAULT_PROCESSING
        p=project(pid);state=p.state();image=p.image(state,body['image_id'])
        return plan_tiles(image['width'],image['height'],body.get('tiling') or state.get('processing_defaults') or DEFAULT_PROCESSING,image_sha256=image['normalized_sha256'])
    @api.post('/projects/{pid}/processing')
    def processing_defaults(pid:str,body:dict):
        from compag_annotator.geometry import plan_tiles
        expected(body);p=project(pid);tiling=body['tiling'];plan_tiles(1,1,tiling)
        with p.edit(body['expected_revision'],'processing_defaults',service.actor) as (db,state):
            state['processing_defaults']=tiling
        return p.public()
    @api.get('/projects/{pid}/generation_layers')
    def generation_layers(pid:str):return project(pid).state().get('generation_layers',[])
    @api.post('/projects/{pid}/generate')
    def generate(pid:str,body:dict,request:Request):
        expected(body);return service.submit_generation(pid,body,request.headers.get('idempotency-key'))
    @api.post('/projects/{pid}/selection')
    def selection(pid:str,body:dict):
        from compag_annotator.core.selection import apply_selection
        expected(body);return apply_selection(project(pid),body,service.actor)
    @api.post('/projects/{pid}/merge/preview')
    def merge_preview(pid:str,body:dict):
        from compag_annotator.core.selection import preview_merge
        expected(body);return preview_merge(project(pid),body)
    @api.post('/projects/{pid}/merge')
    def merge_masks(pid:str,body:dict):
        from compag_annotator.core.selection import merge_selected
        expected(body);return merge_selected(project(pid),body,service.actor)
    @api.post('/projects/{pid}/boundary/annotation')
    def annotation_boundary(pid:str,body:dict,request:Request):
        expected(body);return service.submit_annotation_boundary(pid,body,request.headers.get('idempotency-key'))
    @api.post('/projects/{pid}/images/upload')
    async def upload_images(pid:str,files:list[UploadFile]=File(...)):
        project(pid);paths=await save_uploads(files)
        names={p:json.loads((Path(p).parent/'upload_name.json').read_text())['name'] for p in paths}
        return submit('import_images',{'project_id':pid,'paths':paths,'storage':'copy','display_names':names,'uploaded':True})
    @api.post('/projects/{pid}/images/import')
    def import_images(pid:str,body:dict,request:Request):
        project(pid)
        if body.get('approved') is not True:raise ValueError('Explicitly approve the chosen local import path')
        path=Path(body['path']).expanduser().resolve(strict=True)
        if path.is_dir():paths=path.rglob('*') if body.get('recursive') is True else path.iterdir();files=[str(p) for p in paths if p.is_file() and p.resolve().is_relative_to(path)]
        elif path.is_file():files=[str(path)]
        else:raise ValueError('Choose an image file or directory')
        if len(files)>50000:raise ValueError('Import batch exceeds 50,000 files')
        return submit('import_images',{'project_id':pid,'paths':files,'storage':body.get('storage','copy')},request)
    @api.get('/projects/{pid}/images/{iid}/media')
    def media(pid:str,iid:str):
        p=project(pid);image=p.image(p.state(),iid);path=safe_child(p.path,image['path'])
        if not path.exists():raise ValueError('Image is missing; relink its source')
        return FileResponse(path,media_type='image/png')
    @api.get('/projects/{pid}/images/{iid}/thumbnail')
    def thumbnail(pid:str,iid:str):
        p=project(pid);image=p.image(p.state(),iid);return FileResponse(safe_child(p.path,image['thumbnail']),media_type='image/jpeg')
    @api.patch('/projects/{pid}/images/{iid}')
    def image_update(pid:str,iid:str,body:dict):expected(body);return project(pid).update_image(iid,body,service.actor)
    @api.post('/projects/{pid}/images/{iid}/complete')
    def image_complete(pid:str,iid:str,body:dict):expected(body);return project(pid).update_image(iid,body,service.actor)
    @api.post('/projects/{pid}/images/{iid}/review')
    def image_review(pid:str,iid:str,body:dict):
        from compag_annotator.core.image_review import confirm_image_review
        expected(body);return confirm_image_review(project(pid),iid,body,service.actor)
    @api.delete('/projects/{pid}/images/{iid}')
    def remove_image(pid:str,iid:str,body:dict):
        expected(body)
        with service.jobs.lock:
            if any(j['status'] in ('queued','running') and j.get('payload',{}).get('project_id')==pid for j in service.jobs.list()):
                raise ValueError('Wait for this project’s current jobs to finish, or cancel them in Jobs, before removing an image')
            return project(pid).remove_image(iid,body,service.actor)
    @api.post('/projects/{pid}/images/{iid}/relink')
    def relink(pid:str,iid:str,body:dict):
        if body.get('approved') is not True:raise ValueError('Approve the replacement local path')
        p=project(pid);path=Path(body['path']).expanduser().resolve(strict=True)
        with p.edit(body.get('expected_revision'),'image_relink',service.actor) as (db,state):
            image=p.image(state,iid)
            if digest(path)!=image['sha256']:raise ValueError('Replacement is not the same original image bytes')
            image['original']=str(path);image['storage']='reference'
            # Missing managed display can be reconstructed only from this same source.
            if not safe_child(p.path,image['path']).exists():
                from PIL import Image,ImageOps
                target=safe_child(p.path,image['path']);target.parent.mkdir(parents=True,exist_ok=True)
                with Image.open(path) as im:ImageOps.exif_transpose(im).convert('RGB').save(target)
                if digest(target)!=image['normalized_sha256']:raise ValueError('Normalized image differs; inspect decoder compatibility')
        return p.public()
    @api.get('/projects/{pid}/images/{iid}/annotations')
    def annotations(pid:str,iid:str):
        from collections import Counter
        p=project(pid);p.image(p.state(),iid);rows=p.annotations(iid)
        counts=dict(Counter(row['status'] for row in rows));counts['unassigned']=sum(row['class_id'] is None and row['status'] not in ('rejected','superseded') for row in rows)
        return {'annotations':rows,'revision':p.state()['revision'],'counts':counts}
    @api.get('/projects/{pid}/annotations/{aid}/geometry')
    def geometry(pid:str,aid:str):return project(pid).get_annotation(aid)['geometry']
    @api.post('/projects/{pid}/images/{iid}/annotations')
    def add_annotation(pid:str,iid:str,body:dict):expected(body);return project(pid).annotate(iid,body,service.actor)
    @api.patch('/projects/{pid}/annotations/{aid}')
    def update_annotation(pid:str,aid:str,body:dict):
        expected(body);p=project(pid);row=p.get_annotation(aid);return p.annotate(row['image_id'],body,service.actor,aid)
    @api.delete('/projects/{pid}/annotations/{aid}')
    def delete_annotation(pid:str,aid:str,body:dict):
        expected(body);p=project(pid);row=p.get_annotation(aid);return p.annotate(row['image_id'],body,service.actor,aid,delete=True)
    @api.post('/projects/{pid}/images/{iid}/history')
    def history(pid:str,iid:str,body:dict):expected(body);return project(pid).history(iid,body,service.actor)
    @api.post('/projects/{pid}/rounds/preview')
    def rounds_preview(pid:str,body:dict):return project(pid).plan_rounds(body,service.actor,True)
    @api.post('/projects/{pid}/rounds')
    def rounds_commit(pid:str,body:dict):expected(body);return project(pid).plan_rounds(body,service.actor)
    @api.post('/projects/{pid}/rounds/{rid}')
    def round_action(pid:str,rid:str,body:dict):expected(body);return project(pid).round_action(rid,body,service.actor)
    @api.get('/models/catalog')
    def model_catalog():return service.manager.catalog()
    @api.get('/models/status')
    def model_status():return service.manager.status()
    @api.get('/models/devices')
    def model_devices(provider:str='sam2',refresh:bool=False):return service.manager.devices(provider,refresh=refresh)
    @api.get('/models')
    def models():return service.manager.models()
    @api.post('/models/install')
    def install_model(body:dict,request:Request):
        if body.get('consent') is not True:raise ValueError('Review source/license and explicitly consent to installation')
        if body.get('provider')=='sam3' and body.get('download_weights'):raise ValueError('SAM3 checkpoint downloads are disabled; select local weights')
        return submit('install_model',body,request)
    @api.post('/models/register')
    def register_model(body:dict):
        if body.get('approved') is not True:raise ValueError('Approve the local checkpoint path')
        return service.manager.register(body['provider'],body['path'],architecture=body.get('architecture'),trust=body.get('trust') is True,class_mapping=body.get('class_mapping'))
    @api.post('/models/{mid}/test')
    def test_model(mid:str,body:dict,request:Request):return submit('test_model',{**body,'model_id':mid},request)
    @api.post('/models/unload')
    def unload():return service.manager.unload()
    @api.post('/projects/{pid}/models/activate')
    def activate(pid:str,body:dict):expected(body);return service.activate(project(pid),body)
    @api.post('/projects/{pid}/models/rollback')
    def rollback(pid:str,body:dict):expected(body);return service.rollback(project(pid),body)
    @api.post('/projects/{pid}/images/{iid}/assist')
    def assist(pid:str,iid:str,body:dict,request:Request):
        expected(body);return submit('assist',{**body,'project_id':pid,'image_id':iid},request)
    @api.post('/projects/{pid}/images/{iid}/assist/{jid}/apply')
    def apply_assist(pid:str,iid:str,jid:str,body:dict):
        from compag_annotator.core.assist import apply_independent_preview
        expected(body);return apply_independent_preview(service,pid,iid,jid,body)
    @api.post('/projects/{pid}/predict')
    def predict(pid:str,body:dict,request:Request):expected(body);return submit('predict',{**body,'project_id':pid},request)
    @api.get('/projects/{pid}/training/readiness')
    def ready(pid:str,round_id:str|None=None,allow_lossy:bool=False,training_mode:str='whole'):
        return readiness(project(pid),qa_smoke=service.qa_mode,round_id=round_id,allow_lossy=allow_lossy,training_mode=training_mode)
    @api.post('/projects/{pid}/training/exclusions')
    def training_exclusions(pid:str,body:dict):
        from compag_annotator.training.exclusions import change_training_exclusions
        expected(body);return change_training_exclusions(project(pid),body,service.actor)
    @api.post('/projects/{pid}/train')
    def train(pid:str,body:dict,request:Request):return submit('train',{**body,'project_id':pid},request)
    @api.get('/projects/{pid}/boundary/proposals')
    def boundary_proposals(pid:str):return project(pid).state()['boundary_proposals']
    @api.post('/projects/{pid}/boundary/review')
    def boundary_review(pid:str,body:dict):expected(body);return service.boundary.review(project(pid),body,service.actor)
    @api.post('/projects/{pid}/exports')
    def export(pid:str,body:dict,request:Request):return submit('export',{**body,'project_id':pid},request)
    @api.post('/projects/{pid}/exports/preview')
    def export_preview(pid:str,body:dict):
        from compag_annotator.core.dataset_export import plan_dataset
        return plan_dataset(project(pid),body)[0]
    @api.post('/projects/{pid}/imports')
    async def import_annotations(pid:str,file:UploadFile=File(...),format:str=Form(...),options:str=Form('{}')):
        paths=await save_uploads([file]);return submit('import_annotations',{'project_id':pid,'path':paths[0],'format':format,'options':json.loads(options)})
    @api.get('/jobs')
    def jobs():return service.jobs.list()
    @api.get('/jobs/{jid}')
    def job(jid:str):return service.jobs.get(jid)
    @api.post('/jobs/{jid}/cancel')
    def cancel(jid:str):return service.jobs.cancel(jid)
    @api.post('/jobs/{jid}/retry')
    def retry(jid:str):
        previous=service.jobs.get(jid)
        if previous['status'] not in ('failed','cancelled','interrupted'):raise ValueError('Only failed, interrupted or cancelled jobs can be retried')
        return submit(previous['kind'],previous['payload'])
    @api.get('/jobs/{jid}/artifact')
    def artifact(jid:str):
        job=service.jobs.get(jid)
        if job['status']!='complete' or not (job.get('result') or {}).get('artifact'):raise ValueError('No completed artifact')
        path=Path(job['result']['artifact']).resolve()
        if not path.is_relative_to((service.jobs.root/jid).resolve()) or not path.is_file():raise ValueError('Invalid artifact path')
        return FileResponse(path,filename=path.name)
    @api.get('/doctor')
    def doctor():
        return {'version':__version__,'python':platform.python_version(),'platform':platform.system(),'core_ready':True,'qa_mode':service.qa_mode,'free_disk_bytes':shutil.disk_usage(service.data_dir).free,'providers':service.manager.doctor(),'boundary_invocations':service.boundary.invocations}
    @api.post('/diagnostics')
    def diagnostics():return submit('diagnostics',{})
    return api
