"""Immutable cumulative segmentation snapshots; no research label assumptions."""
import copy,json,shutil
from pathlib import Path
import yaml
from compag_annotator.storage.files import atomic,digest,now
from compag_annotator.training.guidance import explain_readiness,readiness_message

def readiness(project,*,qa_smoke=False,round_id=None,allow_lossy=False,document=None,training_mode='whole',_tiles=None):
    from compag_annotator.geometry import polygon_for_yolo
    from compag_annotator.training.tiles import training_layout,tile_labels
    layout=training_layout(training_mode);compiled={} if _tiles is None else _tiles
    document=project.document(reviewed_only=False) if document is None else document;classes=[c for c in document['classes'] if not c['archived']]
    mapping={c['id']:i for i,c in enumerate(sorted(classes,key=lambda c:c['order']))}
    class_ids={c['id'] for c in classes};errors=[];warnings=[];excluded=[];included=[];counts={c['id']:0 for c in classes}
    allowed=None
    if round_id:
        selected=next((r for r in document['rounds'] if r['id']==round_id),None)
        if selected is None:raise ValueError('Round not found')
        allowed={i for r in document['rounds'] if r['number']<=selected['number'] for i in r['image_ids']}
    for image in document['images']:
        reason=None
        if image.get('training_excluded'):reason='explicit_training_exclusion'
        elif image['role'] not in ('pool','train','validation'):reason='role_excluded'
        elif allowed is not None and image['role']=='pool' and image['id'] not in allowed:reason='future_round'
        elif image.get('storage')=='reference' and not Path(image['original']).is_file():reason='linked_original_missing'
        elif not Path(image['path']).is_file():reason='image_missing'
        elif not image['complete']:reason='image_incomplete'
        elif image['review_actor']!='human' and not (qa_smoke and image['review_actor']=='automated_qa'):reason='awaiting_human_review'
        if reason:excluded.append({'image_id':image['id'],'reason':reason});continue
        accepted=[r for r in document['annotations'] if r['image_id']==image['id'] and r['status']=='accepted']
        rows=[r for r in accepted if not r.get('training_excluded')]
        if accepted and not rows:
            excluded.append({'image_id':image['id'],'reason':'all_instances_excluded'});continue
        for row in rows:
            if row['class_id'] not in class_ids:errors.append({'image_id':image['id'],'annotation_id':row['id'],'reason':'unassigned_or_archived_class'});continue
            if row.get('review_actor')!='human' and not (qa_smoke and row.get('review_actor')=='automated_qa'):errors.append({'annotation_id':row['id'],'reason':'annotation_not_human_reviewed'})
            if row['geometry']['type']=='box':errors.append({'annotation_id':row['id'],'reason':'box_only_not_segmentation_ground_truth'});continue
            if training_mode=='whole':
                try:polygon_for_yolo(row['geometry'],image['width'],image['height'],allow_lossy=allow_lossy)
                except ValueError as e:errors.append({'annotation_id':row['id'],'reason':str(e)})
            counts[row['class_id']]+=1
        if training_mode=='tiles_512':
            compiled[image['id']]=tile_labels(image,[r for r in rows if r['class_id'] in class_ids and r['geometry']['type']!='box'],mapping,allow_lossy=allow_lossy)
            errors.extend(compiled[image['id']]['errors'])
        included.append(image)
    train=[i for i in included if i['role'] in ('pool','train')];val=[i for i in included if i['role']=='validation']
    if not classes:errors.append({'reason':'Create at least one object class'})
    if not train:errors.append({'reason':'No complete reviewed training images'})
    if not val:errors.append({'reason':'Supply separately reviewed fixed validation images; training is not reused as validation'})
    train_ids={i['id'] for i in train};train_groups={i['group_id'] for i in train};val_groups={i['group_id'] for i in val}
    if train_groups&val_groups:errors.append({'reason':'Related groups span training and validation'})
    training_objects=[r for r in document['annotations'] if r['image_id'] in train_ids and r['status']=='accepted' and not r.get('training_excluded')]
    if not training_objects:errors.append({'reason':'Training needs at least one reviewed instance, not only negative images'})
    for c in classes:
        if not any(r['class_id']==c['id'] for r in training_objects):warnings.append('No training instances for class '+c['name'])
    report={'ready':not errors,'errors':errors,'warnings':warnings,'training_images':len(train),'validation_images':len(val),'instances_per_class':counts,'negative_images':sum(not any(r['image_id']==i['id'] and r['status']=='accepted' for r in document['annotations']) for i in included),'included_image_ids':[i['id'] for i in included],'excluded':excluded,'qa_smoke':qa_smoke,'human_uat':False if qa_smoke else None,'generalization_claim':False}
    report['training_instances_per_class']={c['id']:sum(r['class_id']==c['id'] for r in training_objects) for c in classes}
    report['project_revision']=document['revision']
    report['training_layout']=layout
    if training_mode=='tiles_512':
        report['training_tiles']=sum(len(compiled[i['id']]['tiles']) for i in train)
        report['validation_tiles']=sum(len(compiled[i['id']]['tiles']) for i in val)
        report['training_tile_instances']=sum(len(t['lines']) for i in train for t in compiled[i['id']]['tiles'])
        report['validation_tile_instances']=sum(len(t['lines']) for i in val for t in compiled[i['id']]['tiles'])
    report['excluded_annotations']=[{'annotation_id':r['id'],'annotation_revision':r['revision'],
        'image_id':r['image_id'],'class_id':r['class_id'],'status':r['status'],
        'reason':r.get('training_exclusion',{}).get('reason','explicit_training_exclusion')}
        for r in document['annotations'] if r.get('training_excluded')]
    return explain_readiness(report,document)

def build_snapshot(project,directory,settings):
    from compag_annotator.geometry import polygon_for_yolo,validate_yolo_rows
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    mode=settings.get('training_mode','whole');compiled={}
    if mode=='tiles_512' and settings.get('imgsz',512)!=512:raise ValueError('512 x 512 tile training requires Image size 512 for matching inference')
    with project.lock:
        doc=project.document(reviewed_only=False)
        check=readiness(project,qa_smoke=settings.get('qa_smoke',False),round_id=settings.get('round_id'),allow_lossy=settings.get('allow_lossy',False),document=doc,training_mode=mode,_tiles=compiled)
        if not check['ready']:raise ValueError(readiness_message(check))
    classes=sorted([c for c in doc['classes'] if not c['archived']],key=lambda c:c['order']);mapping={c['id']:i for i,c in enumerate(classes)}
    ids=set(check['included_image_ids']);included=[i for i in doc['images'] if i['id'] in ids];conversions=[];file_hashes={};tile_records=[]
    for image in included:
        split='val' if image['role']=='validation' else 'train'
        if mode=='tiles_512':
            from compag_annotator.training.tiles import write_tiles
            tile_records.extend(write_tiles(image,compiled[image['id']],directory,split,file_hashes))
            conversions.extend(f for t in compiled[image['id']]['tiles'] for f in t['fragments'])
            continue
        imdir=directory/'images'/split;labeldir=directory/'labels'/split;imdir.mkdir(parents=True,exist_ok=True);labeldir.mkdir(parents=True,exist_ok=True)
        target=imdir/(image['id']+Path(image['path']).suffix);shutil.copy2(image['path'],target)
        if digest(target)!=image['normalized_sha256']:raise ValueError('Image derivative changed during snapshot')
        lines=[]
        for row in doc['annotations']:
            if row['image_id']!=image['id'] or row['status']!='accepted' or row.get('training_excluded'):continue
            points,report=polygon_for_yolo(row['geometry'],image['width'],image['height'],allow_lossy=settings.get('allow_lossy',False))
            points=[(float(x)/image['width'],float(y)/image['height']) for x,y in points]
            lines.append(str(mapping[row['class_id']])+' '+' '.join(format(v,'.17g') for point in points for v in point))
            conversions.append({'annotation_id':row['id'],'revision':row['revision'],**report})
        validate_yolo_rows(lines)
        labels=labeldir/(image['id']+'.txt');labels.write_text('\n'.join(lines)+ ('\n' if lines else ''))
        file_hashes[str(target.relative_to(directory))]=digest(target);file_hashes[str(labels.relative_to(directory))]=digest(labels)
    data={'path':str(directory.resolve()),'train':'images/train','val':'images/val','names':{i:c['name'] for i,c in enumerate(classes)}}
    (directory/'data.yaml').write_text(yaml.safe_dump(data,allow_unicode=True,sort_keys=False))
    receipt={'schema':'compag-training-snapshot/v1','created_at':now(),'project_revision':doc['revision'],'class_version':doc['class_version'],'class_mapping':{str(i):c['id'] for i,c in enumerate(classes)},'classes':classes,'image_ids':[i['id'] for i in included],'train_image_ids':[i['id'] for i in included if i['role']!='validation'],'validation_image_ids':[i['id'] for i in included if i['role']=='validation'],'image_groups':{i['id']:i['group_id'] for i in included},'file_hashes':file_hashes,'annotation_versions':[{k:r[k] for k in ('id','image_id','class_id','revision','review_actor','human_verified')} for r in doc['annotations'] if r['image_id'] in ids and r['status']=='accepted' and not r.get('training_excluded')],'settings':settings,'readiness':check,'conversion_report':conversions,'boundary_policy':doc['boundary_policy'],'boundary_recovery_requested':settings.get('boundary_opt_in',False)}
    from compag_annotator.config import BOUNDARY_POLICY
    receipt['effective_boundary_policy']=copy.deepcopy(BOUNDARY_POLICY)
    receipt['excluded_annotations']=copy.deepcopy(check['excluded_annotations'])
    receipt['training_layout']=copy.deepcopy(check['training_layout'])
    if mode=='tiles_512':receipt['tiles']=tile_records
    atomic(directory/'snapshot.json',receipt)
    return {'dataset_yaml':str(directory/'data.yaml'),'receipt':receipt,'snapshot_sha256':digest(directory/'snapshot.json')}
