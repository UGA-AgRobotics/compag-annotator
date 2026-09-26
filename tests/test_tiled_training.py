"""Exact crops, strict segmentation labels, source-group splits and model recipes."""
import copy
import json
import numpy as np
import pytest
from PIL import Image
from compag_annotator.core.projects import Project
from compag_annotator.geometry import encode_rle, geometry_mask, map_tile_geometry
from compag_annotator.training.tiles import padded_plan, crop_rle, training_layout, resolve_prediction_settings
from compag_annotator.training.snapshots import readiness, build_snapshot
from compag_annotator.geometry.stitching import stitch_predictions
from compag_annotator.providers.yolo import validate_tiled_prediction, train_settings


def test_exact_crop_mapping_randomized_and_edge_coverage():
    for w,h in [(70,64),(1024,1024),(1100,700),(3024,4032)]:
        plan=padded_plan(w,h)
        cover=np.zeros((h,w),bool)
        rng=np.random.default_rng(42)
        mask=rng.random((h,w))>.99
        reconstructed=np.zeros_like(mask)
        for tile in plan['tiles']:
            x0,y0,x1,y1=tile['box'];cover[y0:y1,x0:x1]=True
            geometry=crop_rle({'size':[h,w],'counts':encode_rle(mask)['counts']},tile['box'])
            if geometry:
                local=geometry_mask(geometry,512,512)
                np.testing.assert_array_equal(local[:y1-y0,:x1-x0],mask[y0:y1,x0:x1])
                assert not local[y1-y0:,:].any() and not local[:,x1-x0:].any()
                reconstructed |= geometry_mask(map_tile_geometry(geometry,tile,w,h),w,h)
        assert cover.all()
        np.testing.assert_array_equal(mask,reconstructed)
    assert [t['box'][0] for t in padded_plan(1100,512)['tiles']]==[0,512,588]
    assert len(padded_plan(3024,4032)['tiles'])==48


def make_project(tmp_path):
    p=Project.create(tmp_path/'p','Tile QA',[{'name':'Any class'}])
    for index,role in enumerate(('train','validation')):
        # Distinct real files, deterministic noise avoids identical negative tile bytes.
        path=tmp_path/f'{role}.png'
        pixels=np.random.default_rng(index).integers(0,256,(600,1100,3),dtype=np.uint8)
        Image.fromarray(pixels).save(path);p.add_images([path]);iid=p.state()['images'][-1]['id']
        p.update_image(iid,{'role':role},'automated_qa')
        p.annotate(iid,{'class_id':p.state()['classes'][0]['id'],'status':'accepted',
                       'geometry':{'type':'polygon','points':[[500,250],[530,250],[530,280],[500,280]]}},'automated_qa')
        p.update_image(iid,{'complete':True,'attest':True},'automated_qa')
    return p


def test_snapshot_tile_labels_pixel_exact_and_originals_preserved(tmp_path):
    p=make_project(tmp_path);before=copy.deepcopy(p.document(reviewed_only=False))
    report=readiness(p,qa_smoke=True,training_mode='tiles_512')
    assert report['ready'] and report['training_tiles']==report['validation_tiles']==6
    assert report['training_tile_instances']==4
    result=build_snapshot(p,tmp_path/'snapshot',{'training_mode':'tiles_512','imgsz':512,'qa_smoke':True})
    receipt=result['receipt'];assert len(receipt['tiles'])==12
    assert receipt['training_layout']['metric_scope']=='tile'
    doc=p.document(reviewed_only=False)
    for tile in receipt['tiles']:
        src=next(i for i in doc['images'] if i['id']==tile['image_id'])
        assert tile['split']==('val' if src['role']=='validation' else 'train')
        assert tile['group_id']==src['group_id']
        file=tmp_path/'snapshot'/tile['image_file'];assert Image.open(file).size==(512,512)
        x0,y0,x1,y1=tile['box']
        np.testing.assert_array_equal(np.asarray(Image.open(file)),np.asarray(Image.open(src['path']))[y0:y1,x0:x1])
        lines=(tmp_path/'snapshot'/tile['label_file']).read_text().splitlines()
        assert len(lines)==tile['instances']
        if lines:
            vals=list(map(float,lines[0].split()[1:]));coords=np.array(vals).reshape(-1,2)*512
            decoded=geometry_mask({'type':'polygon','points':coords.tolist()},512,512)
            original=geometry_mask(next(a for a in doc['annotations'] if a['image_id']==tile['image_id'])['geometry'],1100,600)
            np.testing.assert_array_equal(decoded,original[y0:y1,x0:x1])
    assert before==doc
    from compag_annotator.providers.yolo import local_dataset
    _,_,inventory=local_dataset(result['dataset_yaml']);assert len(inventory['train'])==6
    frozen=json.loads((tmp_path/'snapshot/snapshot.json').read_text())
    assert len(frozen['annotation_versions'])==2  # Original instances, not duplicate tile objects.


def test_strict_tile_topology_and_no_cross_group_leakage(tmp_path):
    p=make_project(tmp_path);row=p.annotations()[0];iid=row['image_id']
    mask=np.zeros((600,1100),bool);mask[10:40,10:40]=True;mask[20:30,20:30]=False
    p.annotate(iid,{'status':'accepted','geometry':{'type':'mask','rle':encode_rle(mask)}},'automated_qa',row['id'])
    p.update_image(iid,{'complete':True,'attest':True},'automated_qa')
    report=readiness(p,qa_smoke=True,training_mode='tiles_512')
    assert not report['ready'];assert any(e.get('tile_id') and 'holes' in e['reason'] for e in report['errors'])
    with p.edit(None,'qa_group','automated_qa') as (db,state):state['images'][1]['group_id']=state['images'][0]['group_id']
    assert any('Related groups' in e['reason'] for e in readiness(p,qa_smoke=True,training_mode='tiles_512')['errors'])
    with pytest.raises(ValueError,match='Image size 512'):build_snapshot(p,tmp_path/'bad',{'training_mode':'tiles_512','imgsz':640})


def test_model_recipe_and_nms_class_identity_and_legacy():
    model={'training_layout':training_layout('tiles_512')}
    values=resolve_prediction_settings(model,{'confidence':.17,'imgsz':640,'tile_size':512})
    assert values['confidence']==.17 and values['imgsz']==512 and 'tile_size' not in values
    plan=validate_tiled_prediction(values,80,70,None);assert plan['tiles'][0]['padding']==[0,0,432,442]
    assert resolve_prediction_settings({}, {'imgsz':640})=={'imgsz':640}
    assert resolve_prediction_settings(model,{'use_training_layout':False,'imgsz':640})=={'imgsz':640}
    with pytest.raises(ValueError):train_settings({'training_layout':model['training_layout'],'imgsz':640})
    with pytest.raises(ValueError):validate_tiled_prediction({**values,'imgsz':640},80,70,None)
    rows=[{'bbox':[500,10,530,40],'score':score,'model_class_index':cls,'source':{'tile_id':str(n)}}
          for n,(score,cls) in enumerate([(.8,0),(.9,0),(.7,1)])]
    result,report=stitch_predictions(rows)
    assert [r['source']['tile_id'] for r in result]==['1','2']
    assert report['suppressed_count']==1 and rows[0]['bbox']==[500,10,530,40]
