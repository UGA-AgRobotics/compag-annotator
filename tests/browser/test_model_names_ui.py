"""Real registry/API/UI date labels and activation; structural weights are not ML models."""
import json
from pathlib import Path
import zipfile
from playwright.sync_api import expect
from test_app import api,project,make_image,upload


def test_dated_models_in_list_activation_and_train_predict_selectors(page,tmp_path):
    p=project(page,'Dated models');upload(page,make_image(tmp_path/'models.png'))
    item=next(r for r in api(page,'/api/projects') if r['id']==p['id'])
    data=Path(item['path']).parents[1]
    # Server-created isolated app data only; never the user's model registry.
    assert 'pytest' in str(data) or 'evidence' in str(data)
    path=tmp_path/'best.pt'
    with zipfile.ZipFile(path,'w') as archive:archive.writestr('fixture.txt','Explicit browser fixture; not model weights')
    rows=[]
    for index in range(2):
        rows.append({'id':f'{index+1:08d}-qa-model','name':'best.pt','path':str(path),'provider':'yolo',
            'architecture':'custom-seg','sha256':str(index)*64,'trusted':True,'created_at':1790399625+index,
            'validation_status':'registered_unverified','task':'segmentation',
            'parent_model_id':'qa-base','class_names':{'0':p['classes'][0]['name']},
            'class_mapping':{'0':p['classes'][0]['id']}})
    (data/'models.json').write_text(json.dumps({'schema_version':1,'models':rows}))
    before=(data/'models.json').read_bytes()
    shown=api(page,'/api/models');names=[r['name'] for r in shown]
    assert names[0]!=names[1] and all('2026-09-26' in n for n in names)
    page.get_by_role('navigation').get_by_role('button',name='Models & AI',exact=True).click()
    for name in names:expect(page.get_by_role('heading',name=name,exact=True)).to_be_visible()
    chosen=page.locator('article').filter(has=page.get_by_role('heading',name=names[1],exact=True))
    chosen.get_by_role('button',name='Activate for project',exact=True).click()
    d=page.get_by_role('dialog');expect(d.get_by_test_id('activation-model-name')).to_have_text('Selected model: '+names[1])
    expect(d.get_by_text(shown[1]['saved_time_label'],exact=True)).to_be_visible()
    page.screenshot(path=str(tmp_path/'dated-model-activation.png'),full_page=True)
    d.get_by_role('button',name='Activate model',exact=True).click()
    expect(page.get_by_role('heading',name='Models & AI',exact=True)).to_be_visible()
    assert api(page,f"/api/projects/{p['id']}")['active_model_id']=='00000002-qa-model'
    page.get_by_role('navigation').get_by_role('button',name='Train',exact=True).click()
    select=page.get_by_label('Base / previous segmentation model',exact=True)
    expect(select).to_have_value('00000002-qa-model');assert all(name in select.inner_text() for name in names)
    page.get_by_role('navigation').get_by_role('button',name='Images',exact=True).click()
    page.get_by_role('checkbox',name='Select models.png',exact=True).check()
    page.get_by_role('button',name='Predict selected',exact=True).click()
    select=page.get_by_role('dialog').get_by_label('Model',exact=True)
    expect(select).to_have_value('00000002-qa-model');assert all(name in select.inner_text() for name in names)
    assert (data/'models.json').read_bytes()==before
