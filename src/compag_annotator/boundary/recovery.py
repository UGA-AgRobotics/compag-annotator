"""Bounded contextual recovery adapted from the preserved Standard_Optimized policy.

Independent of research registries, classes and reference labels. SAM supplies
contextual masks; geometry/quality gates only stage suggestions for user review.
"""
import math
import tempfile
import time
from pathlib import Path
from PIL import Image
from compag_annotator.geometry import map_tile_geometry, mask_metadata
from compag_annotator.geometry.rle import geometry_rle, foreground_intervals

POLICY={'schema':'contextual_boundary/v1','crop_sizes':[512,1024,1536],'context_margin':32,
        'max_attempts':3,'sam_only_max_attempts':1,'target_coverage_min':.90,
        'fragment_coverage_min':.90,'fragment_target_fraction_min':.03,
        'area_ratio_min':.8,'area_ratio_max':3.,'predicted_iou_min':.80,
        'stability_min':.88,'neighbour_coverage_max':.10,'duplicate_iou_min':.95,
        'duplicate_area_ratio_max':1.10}


def overlap(a,b):
    aa=sum(a['counts'][1::2]);bb=sum(b['counts'][1::2]);cross=0
    ia=iter(foreground_intervals(a['counts']));ib=iter(foreground_intervals(b['counts']))
    x=next(ia,None);y=next(ib,None)
    while x and y:
        cross+=max(0,min(x[1],y[1])-max(x[0],y[0]))
        if x[1]<y[1]:x=next(ia,None)
        else:y=next(ib,None)
    return {'area_a':aa,'area_b':bb,'intersection':cross,'iou':cross/max(1,aa+bb-cross)}


def seam(row,image):
    tile=row.get('source',{}).get('tile')
    if not isinstance(tile,(list,tuple)) or len(tile)!=4:return False
    box=mask_metadata(row['geometry'],image['width'],image['height'])['bbox']
    x0,y0,x1,y1=tile;l,t,r,b=box
    return ((x0>0 and l<=x0+1) or (y0>0 and t<=y0+1) or
            (x1<image['width'] and r>=x1-1) or (y1<image['height'] and b>=y1-1))


def context_crops(box,width,height,*,yolo):
    needed=max(box[2]-box[0],box[3]-box[1])+2*POLICY['context_margin']
    seen=set();sizes=POLICY['crop_sizes']
    for attempt in range(POLICY['max_attempts'] if yolo else POLICY['sam_only_max_attempts']):
        side=next((s for s in sizes[attempt:] if s>=needed),sizes[-1])
        if side<needed:break
        cw,ch=min(width,side),min(height,side)
        left=max(0,min(width-cw,(box[0]+box[2]-cw)//2))
        top=max(0,min(height-ch,(box[1]+box[3]-ch)//2))
        crop=(left,top,left+cw,top+ch)
        if crop not in seen:yield list(crop)
        seen.add(crop)


def evaluate(candidate,target,neighbours,image,crop,merge_ids):
    w,h=image['width'],image['height'];geometry=candidate['geometry']
    a=geometry_rle(target['geometry'],w,h);b=geometry_rle(geometry,w,h)
    cross=overlap(a,b);coverage=cross['intersection']/cross['area_a'];ratio=cross['area_b']/cross['area_a']
    reasons=[];matched=[]
    if coverage<POLICY['target_coverage_min']:reasons.append('predicted_target_not_preserved')
    if not POLICY['area_ratio_min']<=ratio<=POLICY['area_ratio_max']:reasons.append('unreliable_area_growth')
    for key,threshold in [('predicted_iou','predicted_iou_min'),('stability_score','stability_min')]:
        v=candidate.get(key)
        if type(v) not in (int,float) or not math.isfinite(v) or v<POLICY[threshold]:reasons.append('missing_or_low_native_'+key)
    box=mask_metadata(geometry,w,h)['bbox']
    if any(box[i]==crop[i] for i in range(4)):reasons.append('candidate_still_touches_crop_boundary')
    if box[0]==0 or box[1]==0 or box[2]==w or box[3]==h:reasons.append('physical_image_boundary_unrecoverable')
    for other in neighbours:
        if other['id']==target['id'] or other['status'] in ('rejected','superseded'):continue
        ob=mask_metadata(other['geometry'],w,h)['bbox']
        if box[2]<=ob[0] or ob[2]<=box[0] or box[3]<=ob[1] or ob[3]<=box[1]:continue
        rel=overlap(geometry_rle(other['geometry'],w,h),b)
        cover=rel['intersection']/rel['area_a']
        # Only scoped, unreviewed, same-class seam fragments can share a replacement.
        source=target.get('source',{});other_source=other.get('source',{})
        same_prediction_class=True
        if 'yolo' in (source.get('kind'),other_source.get('kind')):
            same_prediction_class=(source.get('kind')==other_source.get('kind')=='yolo'
                and source.get('model_id')==other_source.get('model_id')
                and type(source.get('model_class_index')) is int
                and source['model_class_index']==other_source.get('model_class_index'))
        eligible=(same_prediction_class and other['id'] in merge_ids and other['status'] in ('draft','proposal')
                  and not other.get('manually_edited') and not other.get('review_actor')
                  and not other.get('human_verified') and not other.get('training_excluded')
                  and other.get('class_id')==target.get('class_id') and seam(other,image))
        if eligible and cover>=POLICY['fragment_coverage_min'] and rel['intersection']/rel['area_b']>=POLICY['fragment_target_fraction_min']:
            matched.append({'id':other['id'],'revision':other['revision']})
        elif cover>POLICY['neighbour_coverage_max']:reasons.append('could_merge_distinct_neighbour:'+other['id'])
    return {'candidate':candidate,'target_coverage':coverage,'area_ratio':ratio,
            'reasons':sorted(set(reasons)),'matched':matched}


def recover(project,row,image,neighbours,manager,model_id,settings,progress,cancel,merge_ids,on_call):
    width,height=image['width'],image['height'];box=mask_metadata(row['geometry'],width,height)['bbox']
    attempts=[];selected=None;result={}
    # Full-image masks stay encoded; temporary crop pixels live on the WSL project drive.
    scratch=project.path/'cache';scratch.mkdir(exist_ok=True)
    with Image.open(image['path']) as source, tempfile.TemporaryDirectory(prefix='boundary-',dir=scratch) as temp:
        for crop in context_crops(box,width,height,yolo=row.get('source',{}).get('kind')=='yolo'):
            if cancel():raise InterruptedError('Boundary recovery cancelled')
            started=time.monotonic();entry={'crop_box':crop,'alternatives':[]};attempts.append(entry)
            local=Path(temp)/'context.png';source.crop(crop).save(local)
            local_box=[box[0]-crop[0],box[1]-crop[1],box[2]-crop[0],box[3]-crop[1]]
            # One real foreground pixel is guaranteed; an eroded-distance anchor is unnecessary for box prompting.
            rle=geometry_rle(row['geometry'],width,height)
            lo,hi=max(foreground_intervals(rle['counts']),key=lambda pair:pair[1]-pair[0])
            anchor=(lo+hi-1)//2;point=[anchor//height-crop[0],anchor%height-crop[1]]
            on_call()
            try:
                result=manager.infer(model_id,str(local),box=local_box,points=[point],labels=[1],
                    device=settings.get('device','cpu'),settings={**(settings.get('provider_settings') or {}),'boundary_quality':True},progress=progress,cancel=cancel)
                descriptor={'box':crop,'padding':[0,0,0,0]}
                for candidate in [*result.get('annotations',[]),*result.get('alternatives',[])]:
                    candidate={**candidate,'geometry':map_tile_geometry(candidate['geometry'],descriptor,width,height)}
                    entry['alternatives'].append(evaluate(candidate,row,neighbours,image,crop,merge_ids))
                suitable=[e for e in entry['alternatives'] if not e['reasons']]
                suitable.sort(key=lambda e:(-e['candidate']['predicted_iou'],-e['candidate']['stability_score']))
                if suitable:
                    first=suitable[0];a=geometry_rle(first['candidate']['geometry'],width,height)
                    coherent=True
                    for candidate in suitable[1:]:
                        rel=overlap(a,geometry_rle(candidate['candidate']['geometry'],width,height))
                        coherent &= rel['iou']>=POLICY['duplicate_iou_min'] and max(rel['area_a'],rel['area_b'])/min(rel['area_a'],rel['area_b'])<=POLICY['duplicate_area_ratio_max']
                    if coherent:selected=first
                    else:entry['reason']='multiple_distinct_suitable_alternatives'
                entry['loaded_checkpoint_sha256']=result.get('loaded_checkpoint_sha256')
                entry['model_timings']=result.get('timings')
            except Exception as error:
                entry.update(error=str(error),elapsed_seconds=time.monotonic()-started)
                error.boundary_attempts=attempts
                raise
            entry['elapsed_seconds']=time.monotonic()-started
            progress({'stage':'boundary_context','attempt':len(attempts),'crop_box':crop,'elapsed_seconds':entry['elapsed_seconds']})
            if selected:break
    return {'selected':selected,'attempts':attempts,'policy':POLICY,
            'loaded_checkpoint_sha256':result.get('loaded_checkpoint_sha256')}
