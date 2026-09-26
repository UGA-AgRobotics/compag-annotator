"""Real small dataset exports; no inference, training or changes to user projects."""
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from compag_annotator.app import create_app
from compag_annotator.web.service import Service
from compag_annotator.core.projects import ConflictError
from compag_annotator.core.dataset_export import plan_dataset
from compag_annotator.geometry import encode_rle


@pytest.fixture
def dataset(tmp_path):
    service=Service(tmp_path/'data',qa_mode=True)
    project=service.catalog.create('Dataset fixture',[{'name':'Leaf'},{'name':'Fruit'}])
    for index in range(4):
        path=tmp_path/f'image-{index}.png';Image.new('RGB',(48,40),(30,70+index*20,110)).save(path)
        project.add_images([path])
    ids=[i['id'] for i in project.state()['images']]
    cid=project.state()['classes'][0]['id']
    def add(i,status,cls=cid,offset=0):
        mask=np.zeros((40,48),np.uint8);mask[5+offset:12+offset,8:18]=1
        project.annotate(ids[i],{'geometry':{'type':'mask','rle':encode_rle(mask)},'status':status,'class_id':cls,'expected_revision':project.state()['revision']},'automated_qa')
    add(0,'accepted')
    project.update_image(ids[0],{'complete':True,'attest':True,'expected_revision':project.state()['revision']},'automated_qa')
    add(1,'draft');add(1,'proposal',None,15);add(1,'rejected',cid,22)
    add(2,'proposal',None)
    add(3,'accepted')
    project.update_image(ids[3],{'role':'excluded','expected_revision':project.state()['revision']},'automated_qa')
    yield service,project,ids
    service.jobs.shutdown()


def test_preview_keeps_reviewed_and_draft_scopes_honest(dataset):
    _,p,ids=dataset;before=p.document(reviewed_only=False)
    report,doc=plan_dataset(p,{'format':'coco'})
    assert report['ready'] and report['image_count']==report['annotation_count']==1
    assert report['draft_count']==0 and doc['images'][0]['id']==ids[0]
    assert report['excluded']['unassigned_or_archived_class_objects']==2
    report,doc=plan_dataset(p,{'format':'coco','annotation_scope':'labeled'})
    assert report['ready'] and report['image_count']==report['annotation_count']==2
    assert report['draft_count']==1 and report['incomplete_image_count']==1
    assert report['excluded']['excluded_role_images']==1
    assert set(i['id'] for i in doc['images'])==set(ids[:2])
    assert all(a['class_id'] for a in doc['annotations'])
    assert before==p.document(reviewed_only=False)


def test_scope_selection_empty_unreviewed_and_stale_preview(dataset):
    _,p,ids=dataset
    report,_=plan_dataset(p,{'format':'coco','scope':'selected','image_ids':[ids[1]]})
    assert not report['ready'] and report['image_count']==0
    report,_=plan_dataset(p,{'format':'coco','scope':'selected','image_ids':[ids[2]],'annotation_scope':'labeled'})
    assert not report['ready']  # Unlabeled image must not become a negative training example.
    with pytest.raises(ValueError,match='Select existing'):
        plan_dataset(p,{'scope':'selected','image_ids':[]})
    with pytest.raises(ValueError,match='review round'):
        plan_dataset(p,{'scope':'round','round_id':'missing'})
    with pytest.raises(ValueError):plan_dataset(p,{'annotation_scope':'anything'})
    with pytest.raises(ConflictError):plan_dataset(p,{'expected_revision':p.state()['revision']-1})


@pytest.mark.parametrize('option', ['include_images', 'allow_lossy'])
def test_string_false_cannot_enable_images_or_geometry_loss(dataset, option):
    _,p,_=dataset
    with pytest.raises(ValueError, match='must be true or false'):
        plan_dataset(p,{option:'false'})


@pytest.mark.parametrize('fmt,variant,allow',[
    ('coco','per_instance',False),('yolo_seg','per_instance',False),('yolo_box','per_instance',True),
    ('voc','per_instance',True),('labelme','per_instance',False),('png','per_instance',False),
    ('png','semantic',True),('png','instance_id',False),
])
def test_downloadable_training_archives_exclude_history(dataset,tmp_path,fmt,variant,allow):
    service,p,_=dataset;before=p.document(reviewed_only=False)
    payload={'project_id':p.state()['id'],'format':fmt,'annotation_scope':'labeled','png_variant':variant,'allow_lossy':allow,'include_images':False}
    summary,_=plan_dataset(p,payload);assert summary['ready']
    payload['expected_revision']=summary['project_revision']
    target=tmp_path/'export';target.mkdir()
    result=service.work('export',payload,target,lambda p:None,lambda:False)
    assert not result['history_included'] and not result['images_included'] and not result['weights_included']
    assert result['summary']['annotation_count']==2
    assert result['archive_bytes']<40_000 and result['archive_bytes']==Path(result['artifact']).stat().st_size
    with zipfile.ZipFile(result['artifact']) as archive:
        names=archive.namelist()
        assert {'README.md','manifest.json','image_index.json','export_summary.json'}<=set(names)
        assert not any(n.endswith(('.sqlite3','.pt','.pth')) or n.startswith('history/') for n in names)
        assert not any(n.startswith('images/') and n.endswith('.png') for n in names)
        assert result['uncompressed_bytes']==sum(i.file_size for i in archive.infolist())
        assert json.loads(archive.read('export_summary.json'))['draft_count']==1
        if fmt=='coco':
            body=json.loads(archive.read('annotations.json'))
            assert len(body['annotations'])==2 and len(body['images'])==2
        if fmt.startswith('yolo'):
            assert len([n for n in names if n.startswith('labels/')])==2
            assert 'dataset.yaml' in names
    assert before==p.document(reviewed_only=False)
    assert service.boundary.invocations==0 and service._manager is None


def test_portable_export_images_and_revision_guard(dataset,tmp_path):
    service,p,_=dataset
    payload={'project_id':p.state()['id'],'format':'coco','annotation_scope':'reviewed','include_images':True,'expected_revision':p.state()['revision']}
    target=tmp_path/'portable';target.mkdir()
    result=service.work('export',payload,target,lambda p:None,lambda:False)
    with zipfile.ZipFile(result['artifact']) as z:
        images=[n for n in z.namelist() if n.startswith('images/') and n.endswith('.png')]
        assert len(images)==1
        with z.open(images[0]) as stream:
            with Image.open(stream) as image:assert image.size==(48,40)
    payload['expected_revision']-=1
    with pytest.raises(ConflictError):service.work('export',payload,tmp_path/'stale',lambda p:None,lambda:False)
    payload.pop('expected_revision')
    with pytest.raises(ValueError,match='Check the export'):service.work('export',payload,tmp_path/'unchecked',lambda p:None,lambda:False)


def test_explicit_conversion_and_existing_splits(dataset):
    _,p,ids=dataset
    p.update_image(ids[1],{'role':'validation','expected_revision':p.state()['revision']},'automated_qa')
    report,_=plan_dataset(p,{'format':'yolo_box','annotation_scope':'labeled'})
    assert not report['ready'] and report['split_counts']=={'train':1,'val':1,'test':0}
    assert plan_dataset(p,{'format':'yolo_box','annotation_scope':'labeled','allow_lossy':True})[0]['ready']
    assert not plan_dataset(p,{'format':'png','png_variant':'semantic'})[0]['ready']


def test_preview_api_is_read_only(dataset):
    service,p,_=dataset
    with TestClient(create_app(service.data_dir,token='test-token',qa_mode=True),base_url='http://localhost') as client:
        before=p.document(reviewed_only=False)
        response=client.post(f"/api/projects/{p.state()['id']}/exports/preview",json={'format':'coco','annotation_scope':'labeled'},headers={'X-Compag-Token':'test-token','Origin':'http://localhost'})
        assert response.status_code==200,response.text
        assert response.json()['annotation_count']==2
        assert before==p.document(reviewed_only=False)
