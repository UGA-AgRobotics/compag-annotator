"""Versioned class-agnostic projects, transactional edits and durable undo.

Extracts the A registry's optimistic-transaction and separate-geometry concepts;
no binary classifier, fixed seed schedule or scientific features are imported.
"""
from __future__ import annotations
import copy,hashlib,json,re,sqlite3,threading
from contextlib import contextmanager
from pathlib import Path
from compag_annotator.config import BOUNDARY_POLICY,DEFAULT_PROCESSING
from compag_annotator.storage.files import uid,now,atomic,json_bytes,safe_child
from compag_annotator.storage.media import import_media
from .rounds import plan

class ConflictError(ValueError):pass

def clean_name(value):
    if not isinstance(value,str) or not value.strip() or len(value)>240 or any(ord(c)<32 for c in value):raise ValueError('Enter a nonempty name up to 240 characters')
    return value.strip()

def label_record(value,order):
    color=value.get('color','#36c8aa')
    if not isinstance(color,str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',color):raise ValueError('Use a six-digit hexadecimal class color')
    shortcut=value.get('shortcut','')
    if not isinstance(shortcut,str) or len(shortcut)>1:raise ValueError('Shortcut must be a single character')
    return {'id':uid(),'name':clean_name(value['name']),'color':color,'shortcut':shortcut,'order':order,'archived':False}

class Project:
    def __init__(self,path):
        self.path=Path(path).expanduser().resolve()
        if not (self.path/'.compag-annotator.json').is_file():raise ValueError('This folder is not a COMPAG Annotator project')
        self.db_path=self.path/'project.sqlite3';self.lock=threading.RLock()
        if not self.db_path.exists():raise ValueError('Project database is missing')
        with self.connection() as db:
            version=db.execute('PRAGMA user_version').fetchone()[0]
            if version!=1:raise ValueError('Unsupported project schema; back up before upgrading')
            if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Project database integrity check failed')
    @classmethod
    def create(cls,path,name,classes):
        path=Path(path).expanduser().resolve()
        if path.exists() and any(path.iterdir()):raise ValueError('Choose an empty project folder')
        path.mkdir(parents=True,exist_ok=True)
        labels=[label_record(v,i) for i,v in enumerate(classes)]
        if len({v['name'].casefold() for v in labels})!=len(labels):raise ValueError('Class names must be unique')
        state={'id':uid(),'name':clean_name(name),'schema_version':1,'revision':0,'class_version':1,'classes':labels,'images':[],'rounds':[],'round_plans':[],'active_model_id':None,'active_model_mapping':{},'active_model_class_version':None,'model_history':[],'models':[],'boundary_policy':dict(BOUNDARY_POLICY),'boundary_proposals':[],'created_at':now(),'updated_at':now()}
        state.update(processing_defaults=copy.deepcopy(DEFAULT_PROCESSING),generation_layers=[],schema_revision=2)
        db=sqlite3.connect(path/'project.sqlite3')
        db.executescript('PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA user_version=1; CREATE TABLE state (id INTEGER PRIMARY KEY,body TEXT NOT NULL); CREATE TABLE annotations(id TEXT PRIMARY KEY,image_id TEXT NOT NULL,body TEXT NOT NULL); CREATE INDEX by_image ON annotations(image_id); CREATE TABLE history(seq INTEGER PRIMARY KEY AUTOINCREMENT,image_id TEXT,action TEXT,actor TEXT,previous TEXT,current TEXT,undone INTEGER NOT NULL DEFAULT 0,created_at TEXT); CREATE TABLE events(seq INTEGER PRIMARY KEY AUTOINCREMENT,action TEXT,actor TEXT,body TEXT,created_at TEXT);')
        db.execute('INSERT INTO state VALUES (1,?)',(json_bytes(state).decode(),));db.commit();db.close()
        atomic(path/'.compag-annotator.json',{'schema_version':1,'id':state['id']})
        return cls(path)
    @contextmanager
    def connection(self):
        db=sqlite3.connect(self.db_path,timeout=30)
        db.execute('PRAGMA foreign_keys=ON');db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA trusted_schema=OFF')
        try:yield db
        finally:db.close()
    def _state(self,db):return json.loads(db.execute('SELECT body FROM state WHERE id=1').fetchone()[0])
    def _save(self,db,state):db.execute('UPDATE state SET body=? WHERE id=1',(json_bytes(state).decode(),))
    @staticmethod
    def image(state,image_id):
        value=next((i for i in state['images'] if i['id']==image_id and not i.get('removed')),None)
        if value is None:raise ValueError('Image not found')
        return value
    def state(self):
        with self.connection() as db:return self._state(db)
    def public(self):
        state=self.state();state['path']=str(self.path)
        state.setdefault('processing_defaults',None)
        state.setdefault('generation_layers',[])
        state['effective_boundary_policy']=copy.deepcopy(BOUNDARY_POLICY)
        for i in state['images']:
            i['media_url']=f"/api/projects/{state['id']}/images/{i['id']}/media";i['thumbnail_url']=f"/api/projects/{state['id']}/images/{i['id']}/thumbnail"
            i['missing']=i['storage']=='reference' and not Path(i['original']).is_file()
        state['images']=[i for i in state['images'] if not i.get('removed')]
        return state
    @contextmanager
    def edit(self,expected,action,actor='human'):
        with self.lock,self.connection() as db:
            db.execute('BEGIN IMMEDIATE');state=self._state(db)
            try:
                if expected is not None and (type(expected) is not int or state['revision']!=expected):raise ConflictError('Project changed. Reload before saving this edit.')
                yield db,state
                state['revision']+=1;state['updated_at']=now();self._save(db,state)
                db.execute('INSERT INTO events(action,actor,body,created_at) VALUES(?,?,?,?)',(action,actor,json_bytes({'revision':state['revision']}).decode(),now()))
                db.commit()
            except BaseException:db.rollback();raise
    def _rows(self,db,image_id=None):
        sql='SELECT body FROM annotations';args=()
        if image_id is not None:sql+=' WHERE image_id=?';args=(image_id,)
        return [json.loads(r[0]) for r in db.execute(sql,args)]
    def geometry(self,row):
        geometry=row['geometry']
        if 'ref' in geometry:return json.loads(safe_child(self.path,geometry['ref']).read_text())
        return copy.deepcopy(geometry)
    def annotations(self,image_id=None,full=False):
        with self.connection() as db:rows=self._rows(db,image_id)
        pid=self.state()['id']
        for r in rows:
            if full:r['geometry']=self.geometry(r)
            else:r['geometry_url']=f"/api/projects/{pid}/annotations/{r['id']}/geometry"
        return rows
    def get_annotation(self,aid,full=True):
        with self.connection() as db:
            r=db.execute('SELECT body FROM annotations WHERE id=?',(aid,)).fetchone()
        if r is None:raise ValueError('Annotation not found')
        r=json.loads(r[0])
        if full:r['geometry']=self.geometry(r)
        return r
    def _put(self,db,row):db.execute('INSERT OR REPLACE INTO annotations VALUES(?,?,?)',(row['id'],row['image_id'],json_bytes(row).decode()))
    def _geometry(self,geometry,image):
        from compag_annotator.geometry import validate_geometry,geometry_bbox,mask_metadata
        valid=validate_geometry(geometry,image['width'],image['height']);valid=geometry if valid is None else valid
        bbox=geometry_bbox(valid,image['width'],image['height'])
        if valid['type']=='mask':
            body=json_bytes(valid);h=hashlib.sha256(body).hexdigest();rel=f'masks/{h}.json';atomic(self.path/rel,body)
            return {'type':'mask','ref':rel,'sha256':h,'bbox':bbox,'area':mask_metadata(valid,image['width'],image['height'])['area'],'size':[image['height'],image['width']]}
        return valid
    @staticmethod
    def suppress_generation(image,row,reason):
        # Deleted/replaced geometry must not return from an overlapping tile on
        # retry. Image-scoped tombstones participate in the same undo snapshot.
        layer=row.get('layer_id');signature=row.get('generation_signature')
        if layer and signature:
            image.setdefault('generation_tombstones',{}).setdefault(layer,{})[signature]={'annotation_id':row['id'],'reason':reason}
    def _history_before(self,db,image,*,state=None):
        state=self._state(db) if state is None else state
        return {'image':copy.deepcopy(image),'annotations':self._rows(db,image['id']),
                'boundary_proposals':copy.deepcopy([p for p in state.get('boundary_proposals',[]) if p['image_id']==image['id']])}
    def _history(self,db,before,image,action,actor,*,state=None):
        db.execute('DELETE FROM history WHERE image_id=? AND undone=1',(image['id'],))
        current=self._history_before(db,image,state=state)
        db.execute('INSERT INTO history(image_id,action,actor,previous,current,created_at) VALUES(?,?,?,?,?,?)',(image['id'],action,actor,json_bytes(before).decode(),json_bytes(current).decode(),now()))
    def class_change(self,body,actor='human',preview=False):
        state=self.state();action=body.get('action');cid=body.get('id')
        active_ids={i['id'] for i in state['images'] if not i.get('removed')}
        rows=[r for r in self.annotations(full=False) if r['image_id'] in active_ids]
        impact={'annotations':sum(r['class_id']==cid for r in rows),'active_model_requires_compatibility_check':bool(state['active_model_id'])}
        if preview:return impact
        with self.edit(body.get('expected_revision'),f'class_{action}',actor) as (db,state):
            labels=state['classes'];label=next((v for v in labels if v['id']==cid),None)
            if action=='create':labels.append(label_record(body,len(labels)))
            elif action=='reorder':
                ids=body.get('ids',[])
                if len(ids)!=len(labels) or set(ids)!={v['id'] for v in labels}:raise ValueError('Class order must include every class once')
                for i,v in enumerate(ids):next(x for x in labels if x['id']==v)['order']=i
                labels.sort(key=lambda v:v['order'])
            elif label is None:raise ValueError('Class not found')
            elif action=='update':
                new=label_record({**label,**body},label['order'])
                label.update({k:new[k] for k in ('name','color','shortcut')})
            elif action=='archive':
                if impact['annotations'] and body.get('confirm') is not True:raise ValueError('Used class: explicitly confirm archive; objects remain preserved and must be remapped before training')
                label['archived']=True
            elif action=='remap':
                target=next((v for v in labels if v['id']==body.get('target_id') and not v['archived']),None)
                if not target or target['id']==cid or body.get('confirm') is not True:raise ValueError('Choose another active class and confirm remapping')
                active_ids={i['id'] for i in state['images'] if not i.get('removed')}
                for row in self._rows(db):
                    if row['class_id']==cid and row['image_id'] in active_ids:
                        row.update(class_id=target['id'],revision=state['revision']+1,status='draft',review_actor=None,human_verified=False);self._put(db,row)
                        self.image(state,row['image_id'])['complete']=False
                label['archived']=True
            else:raise ValueError('Unknown class action')
            if len({v['name'].casefold() for v in labels if not v['archived']})!=sum(not v['archived'] for v in labels):raise ValueError('Active class names must be unique')
            state['class_version']+=1
        return self.public()
    def add_images(self,paths,storage='copy',progress=None,cancel=None,display_names=None):
        result={'imported':[],'duplicates':[],'errors':[],'notes':[]}
        for index,path in enumerate(paths):
            if cancel and cancel():raise InterruptedError('Import cancelled')
            display_name=(display_names or {}).get(str(path),Path(path).name)
            try:
                from compag_annotator.storage.files import digest
                key=digest(path)
                if any(i['sha256']==key and not i.get('removed') for i in self.state()['images']):result['duplicates'].append(display_name);continue
                row=import_media(self.path,path,storage)
                row['name']=(display_names or {}).get(str(path),row['name'])
                with self.edit(None,'image_import','import') as (db,state):
                    if any(i['sha256']==key and not i.get('removed') for i in state['images']):raise ConflictError('Duplicate concurrent import')
                    state['images'].append(row)
                result['imported'].append(row['id'])
                if row.get('import_note'):result['notes'].append({'name':display_name,'message':row['import_note']})
            except (ValueError,OSError) as e:result['errors'].append({'name':display_name,'error':str(e)})
            finally:
                if progress:progress({'stage':'importing','completed':index+1,'total':len(paths)})
        result['storage_bytes']=sum(p.stat().st_size for p in (self.path/'media').rglob('*') if p.is_file()) if (self.path/'media').exists() else 0
        return result
    def remove_image(self,image_id,body,actor='human'):
        if body.get('confirm') is not True:raise ValueError('Confirm removal from project; original source will be kept')
        with self.edit(body.get('expected_revision'),'image_remove',actor) as (db,state):
            image=self.image(state,image_id)
            if any(image_id in r['image_ids'] and r['status']!='planned' for r in state['rounds']):
                raise ValueError('This image belongs to a started review round and is protected to preserve its history')
            image['removed']=True
            # Planned allocations are mutable; never change started round history.
            for round_ in state['rounds']:
                if round_['status']=='planned' and image_id in round_['image_ids']:
                    round_['image_ids'].remove(image_id)
            state['rounds']=[r for r in state['rounds'] if r['image_ids'] or r['status']!='planned']
        return self.public()
    def update_image(self,image_id,body,actor='human'):
        with self.edit(body.get('expected_revision'),'image_update',actor) as (db,state):
            image=self.image(state,image_id)
            if 'role' in body:
                if body['role'] not in ('pool','train','validation','test','excluded'):raise ValueError('Invalid dataset role')
                if any(image_id in r['image_ids'] for r in state['rounds']):raise ValueError('An image in a fixed plan cannot change roles; create a future plan explicitly')
                target_group=body.get('group_id',image['group_id'])
                conflicts=[i for i in state['images'] if i['id']!=image_id and i['group_id']==target_group and i['role']!=body['role'] and not i.get('removed')]
                if conflicts:raise ValueError('Related group images must use the same dataset role')
                image['role']=body['role']
            if 'group_id' in body:
                gid=clean_name(body['group_id'])
                if any(i['id']!=image_id and i['group_id']==gid and i['role']!=image['role'] and not i.get('removed') for i in state['images']):raise ValueError('Related groups cannot span data roles')
                image['group_id']=gid
            if 'complete' in body:
                if type(body['complete']) is not bool:raise ValueError('Complete must be a boolean')
                if body['complete']:
                    if body.get('attest') is not True:raise ValueError('Explicit completeness review is required, including for empty images')
                    if any(r['status'] in ('draft','proposal') for r in self._rows(db,image_id)):raise ValueError('Accept or reject all draft/proposed objects before marking this image complete')
                    image['review_actor']=actor
                image['complete']=body['complete']
            image['revision']=state['revision']+1
        return self.public()
    def annotate(self,image_id,body,actor='human',aid=None,delete=False):
        with self.edit(body.get('expected_revision'),'annotation_delete' if delete else 'annotation_edit',actor) as (db,state):
            image=self.image(state,image_id);before=self._history_before(db,image)
            old=next((r for r in before['annotations'] if r['id']==(aid or body.get('id'))),None)
            if aid and old is None:raise ValueError('Annotation not found')
            if old and old['status']=='superseded':raise ValueError('This parent was superseded; undo the merge before editing it')
            if delete:
                if old:self.suppress_generation(image,old,'deleted')
                db.execute('DELETE FROM annotations WHERE id=?',(aid,))
            else:
                row=copy.deepcopy(old) if old else {'id':uid(),'image_id':image_id,'class_id':None,'status':'draft','source':{'kind':'manual'}}
                if old and ('geometry' in body or 'class_id' in body) and 'status' not in body:row['status']='draft'
                for key in ('class_id','status'):
                    if key in body:row[key]=body[key]
                if row['status'] not in ('draft','proposal','accepted','rejected'):raise ValueError('Invalid review status')
                if row['class_id'] is not None and not any(c['id']==row['class_id'] and not c['archived'] for c in state['classes']):raise ValueError('Choose an active project class')
                if row['status']=='accepted' and row['class_id'] is None:raise ValueError('An accepted instance needs a class')
                if 'geometry' in body:
                    if old:self.suppress_generation(image,old,'geometry_replaced')
                    row['geometry']=self._geometry(body['geometry'],image)
                    from compag_annotator.geometry import mask_metadata
                    row.update(mask_metadata(self.geometry(row),image['width'],image['height']))
                    if old:row.pop('generation_signature',None);row['manually_edited']=True
                elif old is None:raise ValueError('Geometry is required')
                if 'source' in body:
                    source=body['source']
                    if not isinstance(source,dict) or source.get('kind') not in ('manual','sam2','sam3','yolo','import'):raise ValueError('Invalid annotation source')
                    row['source']=copy.deepcopy(source)
                reviewed=row['status']=='accepted'
                if old and row['status']=='rejected':self.suppress_generation(image,old,'rejected')
                row.update(revision=state['revision']+1,review_actor=actor if reviewed else None,human_verified=reviewed and actor=='human')
                self._put(db,row)
            image.update(complete=False,review_actor=None,revision=state['revision']+1)
            self._history(db,before,image,'delete' if delete else 'edit',actor)
        return self.public()
    def history(self,image_id,body,actor='human'):
        action=body.get('action')
        if action not in ('undo','redo'):raise ValueError('Choose undo or redo')
        with self.edit(body.get('expected_revision'),action,actor) as (db,state):
            image=self.image(state,image_id)
            ordering='DESC' if action=='undo' else 'ASC';flag=0 if action=='undo' else 1
            row=db.execute(f'SELECT seq,previous,current FROM history WHERE image_id=? AND undone=? ORDER BY seq {ordering} LIMIT 1',(image_id,flag)).fetchone()
            if row is None:raise ValueError('No action to '+action)
            value=json.loads(row[1 if action=='undo' else 2])
            active_classes={c['id'] for c in state['classes'] if not c['archived']}
            if any(a['class_id'] is not None and a['class_id'] not in active_classes for a in value['annotations']):raise ValueError('This history belongs to an earlier class schema; remap classes explicitly instead of restoring obsolete labels')
            db.execute('DELETE FROM annotations WHERE image_id=?',(image_id,))
            previous_revisions={a['id']:a['revision'] for a in value['annotations']}
            for item in value['annotations']:
                item['revision']=state['revision']+1;self._put(db,item)
            if 'boundary_proposals' in value:
                proposals=copy.deepcopy(value['boundary_proposals'])
                for proposal in proposals:
                    if proposal['annotation_revision']==previous_revisions.get(proposal['annotation_id']):
                        proposal['annotation_revision']=state['revision']+1
                    for parent in proposal.get('matched_annotations',[]):
                        if parent['revision']==previous_revisions.get(parent['id']):
                            parent['revision']=state['revision']+1
                state['boundary_proposals']=[p for p in state.get('boundary_proposals',[]) if p['image_id']!=image_id]+proposals
            restored=value['image'];restored['revision']=state['revision']+1;restored['complete']=False;restored['review_actor']=None
            state['images'][state['images'].index(image)]=restored
            db.execute('UPDATE history SET undone=? WHERE seq=?',(1 if action=='undo' else 0,row[0]))
        return self.public()
    def plan_rounds(self,body,actor='human',preview=False):
        state=self.state();locked=[r for r in state['rounds'] if r['status']!='planned'];locked_ids={i for r in locked for i in r['image_ids']}
        images=[i for i in state['images'] if i['id'] not in locked_ids]
        result=plan(images,body['count'],body.get('order','import'),body.get('seed',42),body.get('image_ids'))
        if preview:return result
        if body.get('confirm') is not True:raise ValueError('Confirm the displayed round plan')
        with self.edit(body.get('expected_revision'),'round_plan',actor) as (db,state):
            if state['rounds']:state['round_plans'].append(copy.deepcopy(state['rounds']))
            for i,r in enumerate(result['rounds']):r['number']=len(locked)+i+1
            state['rounds']=locked+result['rounds']
        return self.public()
    def round_action(self,rid,body,actor='human'):
        with self.edit(body.get('expected_revision'),'round_'+body.get('action',''),actor) as (db,state):
            row=next((r for r in state['rounds'] if r['id']==rid),None)
            if row is None:raise ValueError('Round not found')
            if body['action']=='start':
                if row['status']!='planned':raise ValueError('Round already started/completed')
                if any(r['status']=='in_progress' for r in state['rounds']):raise ValueError('Finish the current round first')
                row.update(status='in_progress',model_id=state['active_model_id'],started_at=now())
            elif body['action']=='finish':
                if row['status']!='in_progress':raise ValueError('Start this round before completion')
                if any(not self.image(state,i)['complete'] for i in row['image_ids']):raise ValueError('Every image needs an explicit completeness review')
                row.update(status='review_complete',training_status='not_run',finished_at=now(),review_actor=actor)
            else:raise ValueError('Unknown round action')
        return self.public()
    def document(self,image_ids=None,reviewed_only=True):
        with self.connection() as db:
            db.execute('BEGIN');state=self._state(db);rows=self._rows(db)
        selected=[i for i in state['images'] if not i.get('removed') and (image_ids is None or i['id'] in image_ids)]
        if reviewed_only:selected=[i for i in selected if i['complete']]
        for i in selected:i['path']=str(safe_child(self.path,i['path']))
        ids={i['id'] for i in selected}
        rows=[r for r in rows if r['image_id'] in ids and r['status'] not in ('rejected','superseded') and (not reviewed_only or r['status']=='accepted')]
        for r in rows:r['geometry']=self.geometry(r)
        return {**state,'images':selected,'annotations':rows,'draft_inclusive':not reviewed_only}
