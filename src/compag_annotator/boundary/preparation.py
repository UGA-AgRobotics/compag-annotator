"""Explicit, scoped, class-agnostic recovery with reviewable replacements."""
from dataclasses import dataclass
import copy
import time
from compag_annotator.storage.files import uid,now
from compag_annotator.jobs.runner import AwaitingReview


@dataclass(frozen=True)
class BoundaryGrant:
    issuer: object
    context: str
    project_id: str
    annotation_ids: tuple
    job_id: str
    limit: int


class BoundaryPreparation:
    def __init__(self):
        self.invocations=0
        self.model_calls=0
        self._issuer=object()

    def authorize(self,project,*,job_kind,job_id,annotation_ids,explicit_opt_in,limit=3):
        # This constructor is called by server-owned job routes. A phase string
        # or a serialized browser object cannot construct a valid grant.
        contexts={'train':'training_preparation','generate':'automatic_mask_annotation','boundary_annotation':'automatic_mask_annotation','predict':'prediction_boundary_check'}
        if job_kind not in contexts or explicit_opt_in is not True:
            raise ValueError('Boundary recovery needs explicit consent for a supported scoped job')
        if type(limit) is not int or not 1<=limit<=20:
            raise ValueError('Boundary limit must be 1–20 candidates')
        if not isinstance(annotation_ids,list) or not all(isinstance(a,str) for a in annotation_ids) or len(set(annotation_ids))!=len(annotation_ids):
            raise ValueError('Boundary scope must contain distinct annotation IDs')
        return BoundaryGrant(self._issuer,contexts[job_kind],project.state()['id'],tuple(annotation_ids),str(job_id),limit)

    def prepare(self,project,manager,settings,progress,cancel,*,authorization=None):
        if (not isinstance(authorization,BoundaryGrant) or authorization.issuer is not self._issuer
                or authorization.project_id!=project.state()['id'] or settings.get('boundary_opt_in') is not True
                or settings.get('phase')!=authorization.context):
            raise ValueError('Heavy boundary recovery requires server-authorized training preparation or automatic-mask annotation scope')
        self.invocations+=1
        from .recovery import seam,recover
        state=project.state();doc=project.document(reviewed_only=authorization.context=='training_preparation')
        scope=set(authorization.annotation_ids)
        rows=[a for a in doc['annotations'] if a['id'] in scope and a['status'] not in ('rejected','superseded')]
        images={i['id']:i for i in doc['images']}
        relevant=[]
        for row in rows:
            if authorization.context=='training_preparation' and row.get('training_excluded'):continue
            if seam(row,images[row['image_id']]):relevant.append(row)
        decided={(p['annotation_id'],p['annotation_revision'],p['model_id']) for p in state['boundary_proposals']
                 if p['status'] in ('retained','excluded','accepted','pending')}
        prior_count=sum((r['id'],r['revision'],settings.get('boundary_model_id')) in decided for r in relevant)
        relevant=[r for r in relevant if (r['id'],r['revision'],settings.get('boundary_model_id')) not in decided]
        receipt={'previous_decisions_respected':True,'previously_decided_or_pending':prior_count,'invoked':True,'context':authorization.context,'job_id':authorization.job_id,
                 'scoped_instances':len(scope),'relevant_instances':len(relevant),'limit':authorization.limit,
                 'model_calls':0,'proposals':0,'deferred_instances':max(0,len(relevant)-authorization.limit)}
        if not relevant:return {**receipt,'applicable':False,'reason':'Previous decisions/pending suggestions already cover the relevant masks' if prior_count else 'No internal tile seam touches these masks; physical image edges are not recoverable seams'}
        model_id=settings.get('boundary_model_id')
        model=next((m for m in manager.models() if m['id']==model_id and m.get('provider') in ('sam2','sam3')),None)
        if model is None:raise ValueError('Choose an installed SAM2/SAM3 model for boundary preparation')
        proposals=[];started=time.monotonic();covered=set()
        for row in relevant[:authorization.limit]:
            if cancel():raise InterruptedError('Boundary preparation cancelled')
            if row['id'] in covered:continue
            existing=next((p for p in state['boundary_proposals'] if p['annotation_id']==row['id'] and p['annotation_revision']==row['revision'] and p['model_id']==model_id and p['status'] in ('retained','excluded','accepted','pending')),None)
            if existing:continue
            image=images[row['image_id']]
            def on_call():
                self.model_calls+=1;receipt['model_calls']+=1
            try:
                result=recover(project,row,image,[a for a in doc['annotations'] if a['image_id']==row['image_id']],
                    manager,model_id,settings,progress,cancel,scope-covered,on_call)
            except Exception as error:
                error.result={**receipt,'error':str(error),'attempts':getattr(error,'boundary_attempts',[]),
                    'recovery_wall_seconds':time.monotonic()-started,'original_masks_preserved':True}
                raise
            selected=result['selected'];matched=selected['matched'] if selected else []
            covered.update([row['id'],*(a['id'] for a in matched)])
            conflicts=sum(any('could_merge_distinct_neighbour:' in reason for reason in e['reasons'])
                          for attempt in result['attempts'] for e in attempt['alternatives'])
            original=row['geometry']
            if matched:
                from compag_annotator.geometry.rle import mask_union
                original=mask_union([row['geometry'],*[a['geometry'] for a in doc['annotations'] if a['id'] in {m['id'] for m in matched}]],image['width'],image['height'])
            proposals.append({'original_geometry':original,'id':uid(),'annotation_id':row['id'],'annotation_revision':row['revision'],
                'image_id':row['image_id'],'class_id':row['class_id'],'model_id':model_id,'status':'pending',
                'geometry':selected['candidate']['geometry'] if selected else None,
                'matched_annotations':matched,'context':authorization.context,'job_id':authorization.job_id,
                'reason':f"Review contextual replacement of {1+len(matched)} object(s); merging requires your acceptance" if selected else
                         'Unresolved boundary: quality, coverage, crop limits or neighbour protection failed. Retain and edit the original explicitly.',
                'source_revision':state['revision'],'created_at':now(),'human_verified':False,
                'neighbour_conflicts':conflicts,'loaded_checkpoint_sha256':result.get('loaded_checkpoint_sha256'),
                'attempts':result['attempts'],'policy':result['policy']})
            progress({'stage':'boundary_preparation','context':authorization.context,'completed':len(proposals),'total':min(len(relevant),authorization.limit),'model_calls':receipt['model_calls']})
        receipt.update(recovery_wall_seconds=time.monotonic()-started,proposals=len(proposals),applicable=True)
        if proposals:
            with project.edit(state['revision'],'boundary_stage','provider') as (db,current):current['boundary_proposals'].extend(proposals)
            error=AwaitingReview('Boundary suggestions are staged. Original masks are unchanged; review each suggestion explicitly.')
            error.result=receipt
            raise error
        return {**receipt,'previous_decisions_respected':True}

    def review(self,project,body,actor):
        if body.get('attest') is not True:raise ValueError('Explicit review required')
        decision=body.get('decision')
        if decision not in ('accept','retain','exclude'):raise ValueError('Choose accept, retain or exclude')
        with project.edit(body.get('expected_revision'),'boundary_review',actor) as (db,state):
            p=next((p for p in state['boundary_proposals'] if p['id']==body['proposal_id']),None)
            if not p or p['status']!='pending':raise ValueError('Pending suggestion not found')
            row=next((a for a in project._rows(db,p['image_id']) if a['id']==p['annotation_id']),None)
            # Removed images remain in project state for history. A stale
            # suggestion for one can still be dismissed, never applied.
            image=next((i for i in state['images'] if i['id']==p['image_id']),None)
            if image is None:raise ValueError('Suggestion image is missing from project history')
            others={a['id']:a for a in project._rows(db,p['image_id'])}
            matched=p.get('matched_annotations',[])
            stale_match=any(a['id'] not in others or others[a['id']]['revision']!=a['revision']
                or others[a['id']]['status'] not in ('draft','proposal') for a in matched)
            stale=(stale_match or row is None or row['revision']!=p['annotation_revision']
                   or row['status'] in ('rejected','superseded') or image.get('removed',False))
            if stale and decision!='retain':
                raise ValueError('Suggestion is stale; choose retain to dismiss it without applying geometry or excluding an image')
            before=project._history_before(db,image,state=state)
            reviewed_at=now()
            if stale:
                p.update(status='dismissed_stale',decision='retain',
                         dismissal_reason='Target annotation or image changed or was removed; stale geometry was not applied',
                         dismissed_annotation_revision=row['revision'] if row else None)
            elif decision=='accept':
                if p['geometry'] is None:raise ValueError('No replacement is available; retain or exclude')
                assigned=any(c['id']==row['class_id'] and not c['archived'] for c in state['classes'])
                project.suppress_generation(image,row,'boundary_accept')
                row['geometry']=project._geometry(p['geometry'],image)
                from compag_annotator.geometry import mask_metadata
                row.update(mask_metadata(p['geometry'],image['width'],image['height']))
                row.pop('generation_signature',None)
                row['manually_edited']=True
                row['source'].setdefault('boundary_reviews',[]).append({'proposal_id':p['id'],'job_id':p.get('job_id'),'model_id':p['model_id'],'model_sha256':p.get('loaded_checkpoint_sha256'),'actor':actor,'reviewed_at':reviewed_at})
                row.update(revision=state['revision']+1,status='accepted' if assigned else 'draft',review_actor=actor if assigned else None,human_verified=assigned and actor=='human',geometry_review_actor=actor);project._put(db,row)
                image.update(complete=False,review_actor=None)
                p['annotation_revision']=row['revision']
                for parent in matched:
                    other=others[parent['id']]
                    project.suppress_generation(image,other,'boundary_merge_accept')
                    other.update(status='superseded',superseded_by=row['id'],revision=state['revision']+1)
                    other['source'].setdefault('boundary_merges',[]).append({'proposal_id':p['id'],'target_id':row['id'],'actor':actor,'reviewed_at':reviewed_at})
                    project._put(db,other)
                    parent['revision']=other['revision']
            elif decision=='exclude':
                image['training_excluded']=True
            if not stale:
                p.update(status={'accept':'accepted','retain':'retained','exclude':'excluded'}[decision],decision=decision)
            p.update(review_actor=actor,reviewed_at=reviewed_at)
            image['revision']=state['revision']+1
            # Include the in-memory decision in the same history entry as the
            # annotation/image change; project.edit commits all of them together.
            project._history(db,before,image,'boundary_'+decision,actor,state=state)
        return project.public()
