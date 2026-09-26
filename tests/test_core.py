import io,json,sqlite3,sys,zipfile
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from compag_annotator.app import create_app
from compag_annotator.core.projects import Project,ConflictError
from compag_annotator.core.catalog import Catalog
from compag_annotator.core.rounds import plan
from compag_annotator.storage.files import safe_extract

def image_file(tmp_path,name='full image ü.png',size=(900,700),color='blue'):
    p=tmp_path/name;Image.new('RGB',size,color).save(p);return p
@pytest.fixture
def project(tmp_path):return Project.create(tmp_path/'project','Arbitrary project',[{'name':'Leaf'},{'name':'Pebble'},{'name':'Third class ✓'}])

def add_image(project,tmp_path,**kw):
    result=project.add_images([image_file(tmp_path,**kw)]);assert len(result['imported'])==1;return result['imported'][0]

def polygon():return {'type':'polygon','points':[[5,5],[80,5],[80,90],[5,90]]}

def annotate(project,iid,geometry=None,status='accepted',actor='automated_qa'):
    state=project.state();return project.annotate(iid,{'geometry':geometry or polygon(),'class_id':state['classes'][0]['id'],'status':status,'expected_revision':state['revision']},actor)

def test_base_import_has_no_ml():
    assert not {'torch','ultralytics','sam2','sam3','xgboost'}&sys.modules.keys()

def test_empty_one_and_multiclass(tmp_path,project):
    assert len(project.state()['classes'])==3
    empty=Project.create(tmp_path/'empty','No model',[]);assert empty.state()['classes']==[]
    one=Project.create(tmp_path/'one','One',[{'name':'Any arbitrary object'}]);assert len(one.state()['classes'])==1

def test_labels_stable_on_rename_reorder_restart(project):
    s=project.state();cid=s['classes'][0]['id'];ids=[c['id'] for c in s['classes']]
    project.class_change({'action':'update','id':cid,'name':'New Unicode name ✓','expected_revision':0})
    project.class_change({'action':'reorder','ids':ids[::-1],'expected_revision':1})
    reloaded=Project(project.path).state();assert reloaded['classes'][-1]['id']==cid;assert reloaded['classes'][-1]['name']=='New Unicode name ✓'

def test_duplicate_class_rejected_without_revision(project):
    with pytest.raises(ValueError):project.class_change({'action':'create','name':'leaf','expected_revision':0})
    assert project.state()['revision']==0

def test_duplicate_import_and_corruption(project,tmp_path):
    p=image_file(tmp_path);duplicate=tmp_path/'same bytes.png';duplicate.write_bytes(p.read_bytes());bad=tmp_path/'bad.png';bad.write_text('not an image')
    r=project.add_images([p,duplicate,bad]);assert len(r['imported'])==1 and len(r['duplicates'])==1 and len(r['errors'])==1
    s=project.state();assert s['images'][0]['width']==900 and s['images'][0]['height']==700
    assert not s['images'][0]['complete']

def test_duplicate_basenames_keep_distinct(project,tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir();p=image_file(a,name='equal.png');q=image_file(b,name='equal.png',color='green')
    assert len(project.add_images([p,q])['imported'])==2

def test_exif_original_preserved(project,tmp_path):
    p=tmp_path/'rotated.jpg';im=Image.new('RGB',(30,20));ex=Image.Exif();ex[274]=6;im.save(p,exif=ex)
    original=p.read_bytes();project.add_images([p]);i=project.state()['images'][0]
    assert (i['width'],i['height'])==(20,30);assert i['exif_orientation']==6
    assert (project.path/i['original']).read_bytes()==original

def test_multipage_and_high_depth_rejected(project,tmp_path):
    p=tmp_path/'multi.tiff';Image.new('RGB',(10,10)).save(p,save_all=True,append_images=[Image.new('RGB',(10,10))]);q=tmp_path/'deep.tiff';Image.new('I;16',(10,10)).save(q)
    assert len(project.add_images([p,q])['errors'])==2

def test_reference_missing_detected(project,tmp_path):
    p=image_file(tmp_path);project.add_images([p],'reference');p.unlink();assert project.public()['images'][0]['missing']

def test_review_and_complete_never_fabricated(project,tmp_path):
    iid=add_image(project,tmp_path);annotate(project,iid,status='proposal')
    with pytest.raises(ValueError):project.update_image(iid,{'complete':True,'attest':True,'expected_revision':2},'automated_qa')
    a=project.annotations(iid)[0];assert not a['human_verified']
    project.annotate(iid,{'status':'accepted','expected_revision':2},'automated_qa',a['id'])
    project.update_image(iid,{'complete':True,'attest':True,'expected_revision':3},'automated_qa')
    assert project.state()['images'][0]['review_actor']=='automated_qa'
    assert not project.annotations(iid)[0]['human_verified']

def test_empty_image_requires_attestation(project,tmp_path):
    iid=add_image(project,tmp_path)
    with pytest.raises(ValueError):project.update_image(iid,{'complete':True,'expected_revision':1})
    project.update_image(iid,{'complete':True,'attest':True,'expected_revision':1})
    assert project.state()['images'][0]['complete']

def test_geometry_undo_redo_delete_restart(project,tmp_path):
    iid=add_image(project,tmp_path);annotate(project,iid);aid=project.annotations(iid)[0]['id']
    project.annotate(iid,{'geometry':{'type':'box','xyxy':[10,20,40,50]},'expected_revision':2},'automated_qa',aid)
    assert project.annotations(iid)[0]['status']=='draft'
    project.history(iid,{'action':'undo','expected_revision':3},'automated_qa');assert project.get_annotation(aid)['geometry']['type']=='polygon'
    p=Project(project.path);p.history(iid,{'action':'redo','expected_revision':4},'automated_qa');assert p.get_annotation(aid)['geometry']['type']=='box'
    p.annotate(iid,{'expected_revision':5},'automated_qa',aid,delete=True);assert not p.annotations(iid)
    p.history(iid,{'action':'undo','expected_revision':6},'automated_qa');assert p.annotations(iid)[0]['id']==aid

def test_stale_revision_no_overwrite(project,tmp_path):
    iid=add_image(project,tmp_path);annotate(project,iid)
    with pytest.raises(ConflictError):project.annotate(iid,{'geometry':polygon(),'class_id':project.state()['classes'][0]['id'],'expected_revision':1})
    assert len(project.annotations(iid))==1

def test_sam_refinement_provenance_survives_restart_and_undo(project,tmp_path):
    iid=add_image(project,tmp_path);annotate(project,iid)
    before=project.annotations(iid)[0]
    source={'kind':'sam2','model_id':'explicit-model-fixture','model_sha256':'fixture-digest'}
    project.annotate(iid,{'geometry':polygon(),'source':source,'expected_revision':2},'automated_qa',before['id'])
    reloaded=Project(project.path);row=reloaded.get_annotation(before['id'])
    assert row['source']==source and row['status']=='draft' and not row['human_verified']
    reloaded.history(iid,{'action':'undo','expected_revision':3},'automated_qa')
    assert reloaded.get_annotation(before['id'])['source']==before['source']

def test_archived_used_class_requires_handling(project,tmp_path):
    iid=add_image(project,tmp_path);annotate(project,iid);s=project.state();cid=s['classes'][0]['id']
    with pytest.raises(ValueError):project.class_change({'action':'archive','id':cid,'expected_revision':s['revision']})
    p=project.class_change({'action':'remap','id':cid,'target_id':s['classes'][1]['id'],'confirm':True,'expected_revision':s['revision']})
    assert project.annotations(iid)[0]['class_id']==s['classes'][1]['id'];assert project.annotations(iid)[0]['status']=='draft'

@pytest.mark.parametrize('n,r,sizes',[(20,3,[7,7,6]),(10,3,[4,3,3]),(1,1,[1]),(6,2,[3,3])])
def test_round_counts(n,r,sizes):
    images=[{'id':str(i),'role':'pool','sha256':str(i)} for i in range(n)]+[{'id':'val','role':'validation','sha256':'val'}]
    p=plan(images,r);assert p['sizes']==sizes;ids=[i for row in p['rounds'] for i in row['image_ids']];assert len(ids)==len(set(ids))==n and 'val' not in ids

@pytest.mark.parametrize('n,r',[(0,1),(1,2),(1,0),(2,True)])
def test_invalid_round_counts(n,r):
    with pytest.raises(ValueError):plan([{'id':str(i),'role':'pool','sha256':str(i)} for i in range(n)],r)

def test_group_split_leak_rejected(project,tmp_path):
    a=add_image(project,tmp_path);b=add_image(project,tmp_path,name='other.png',color='green')
    project.update_image(a,{'role':'validation','expected_revision':2})
    gid=project.state()['images'][0]['group_id']
    with pytest.raises(ValueError):project.update_image(b,{'group_id':gid,'expected_revision':3})

def test_native_exact_backup_restore(project,tmp_path):
    iid=add_image(project,tmp_path);annotate(project,iid);cat=Catalog(tmp_path/'app');cat.register(project)
    before=project.get_annotation(project.annotations(iid)[0]['id']);r=cat.backup(project,tmp_path/'native.zip');restored=cat.restore(r['artifact'])
    assert restored.get_annotation(before['id'])==before;assert restored.state()['classes']==project.state()['classes'];assert restored.state()['id']!=project.state()['id']
    with restored.connection() as db:assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'

@pytest.mark.parametrize('name',['../../escape','/absolute','x/../../../escape','x\\..\\escape'])
def test_archive_paths_rejected(tmp_path,name):
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr(name,b'bad')
    with pytest.raises(ValueError):safe_extract(archive,tmp_path/'out')
    assert not (tmp_path/'out').exists()

def test_api_security_base_manual_import(tmp_path):
    app=create_app(tmp_path/'app',token='qa-secret',qa_mode=True)
    with TestClient(app) as client:
        assert client.get('/api/session').json()['actor']=='automated_qa'
        assert client.post('/api/projects',json={'name':'x'}).status_code==403
        headers={'X-Compag-Token':'qa-secret'}
        assert client.post('/api/projects',json={'name':'x'},headers={**headers,'Origin':'http://evil.invalid'}).status_code==403
        assert client.get('/api/session',headers={'Host':'evil.invalid'}).status_code==403
        created=client.post('/api/projects',json={'name':'<script>alert(1)</script>','classes':[{'name':'Any'}]},headers=headers);assert created.status_code==200
        assert app.state.service.boundary.invocations==0
        pid=created.json()['id'];resp=client.post(f'/api/projects/{pid}/images/import',json={'path':str(tmp_path),'approved':False},headers=headers);assert resp.status_code==400

def test_undo_cannot_reintroduce_archived_class(project,tmp_path):
    iid=add_image(project,tmp_path);annotate(project,iid);a=project.annotations(iid)[0];cid=a['class_id']
    project.annotate(iid,{'geometry':{'type':'box','xyxy':[1,1,20,20]},'expected_revision':2},'automated_qa',a['id'])
    project.class_change({'action':'remap','id':cid,'target_id':project.state()['classes'][1]['id'],'confirm':True,'expected_revision':3})
    with pytest.raises(ValueError,match='earlier class schema'):project.history(iid,{'action':'undo','expected_revision':4})
