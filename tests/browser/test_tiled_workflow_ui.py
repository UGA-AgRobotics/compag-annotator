"""Real browser controls/readiness; provider submissions are explicit UI fixtures."""
from playwright.sync_api import expect
from test_app import api,project,make_image,upload
from test_training_readiness_ui import accepted


def test_training_layout_and_model_bound_prediction(page,tmp_path):
    p=project(page,'Tiled workflow',classes=[{'name':'Any object'}])
    for index in range(2):upload(page,make_image(tmp_path/f'tile-{index}.png',width=1100+index,height=700))
    base=f"/api/projects/{p['id']}";p=api(page,base)
    for index,image in enumerate(p['images']):
        p=accepted(page,p,image['id'])
        p=api(page,base+f"/images/{image['id']}",'PATCH',{'role':'validation' if index else 'train','expected_revision':p['revision']})
        p=api(page,base+f"/images/{image['id']}/complete",'POST',{'complete':True,'attest':True,'expected_revision':p['revision']})
    layout={'mode':'tiled','preset':'512','width':512,'height':512,'overlap':{'mode':'pixels','x':0,'y':0}}
    models=[{'id':'qa-tiled-model','provider':'yolo','name':'QA tile-trained model',
             'training_layout':{'mode':'tiles_512'},'prediction_settings':{'tiling':layout,'imgsz':512}},
            {'id':'qa-old-model','provider':'yolo','name':'QA old model'},
            {'id':'qa-sam','provider':'sam2','name':'QA SAM'}]
    page.route('**/api/models',lambda r:r.fulfill(json=models))
    page.get_by_role('navigation').get_by_role('button',name='Train',exact=True).click()
    expect(page.get_by_label('Training images',exact=True)).to_have_value('tiles_512')
    expect(page.get_by_label('Image size',exact=True)).to_have_value('512')
    assert page.get_by_label('Image size',exact=True).evaluate('(e)=>e.readOnly')
    expect(page.get_by_test_id('training-tile-summary')).to_contain_text('6 training')
    page.get_by_label('Training images',exact=True).select_option('whole')
    expect(page.get_by_test_id('training-tile-summary')).to_have_count(0)
    assert not page.get_by_label('Image size',exact=True).evaluate('(e)=>e.readOnly')
    page.get_by_label('Image size',exact=True).fill('640')
    page.get_by_label('Training images',exact=True).select_option('tiles_512')
    expect(page.get_by_label('Image size',exact=True)).to_have_value('512')
    page.screenshot(path=str(tmp_path/'tile-training-ui.png'),full_page=True)
    page.get_by_role('navigation').get_by_role('button',name='Images',exact=True).click()
    page.get_by_role('checkbox',name='Select tile-0.png',exact=True).check()
    page.get_by_role('button',name='Predict selected',exact=True).click()
    d=page.get_by_role('dialog');expect(d.get_by_test_id('prediction-layout-note')).to_contain_text('Matched to this model')
    expect(d.get_by_label('Model input size',exact=True)).to_have_value('512')
    boundary=d.get_by_role('checkbox',name='Boundary check · slower contextual SAM recovery (optional)',exact=True)
    expect(boundary).not_to_be_checked()
    d.get_by_label('Model',exact=True).select_option('qa-old-model')
    expect(d.get_by_test_id('prediction-layout-note')).to_contain_text('no saved tile-training layout')
    d.get_by_label('Model',exact=True).select_option('qa-tiled-model')
    d.get_by_text('Tiling & inference settings',exact=True).click()
    expect(d.get_by_label('Horizontal overlap',exact=True)).to_have_value('0')
    expect(d.get_by_label('Horizontal overlap',exact=True)).to_be_disabled()
    boundary.check();d.get_by_label('Boundary SAM model',exact=True).select_option('qa-sam')
    d.get_by_label('Maximum boundary objects this job',exact=True).fill('7')
    sent=[]
    def capture(route):
        sent.append(route.request.post_data_json)
        route.fulfill(json={'id':'qa-no-provider-execution','kind':'predict','status':'cancelled','progress':{}})
    page.route('**'+base+'/predict',capture)
    expect(d.get_by_label('Device',exact=True)).to_be_enabled()
    page.screenshot(path=str(tmp_path/'matched-prediction-ui.png'),full_page=True)
    d.get_by_role('button',name='Create proposals',exact=True).click()
    expect(page.get_by_role('heading',name='Jobs',exact=True)).to_be_visible()
    assert sent[0]['settings']['use_training_layout'] is True
    assert sent[0]['settings']['tiling']==layout and sent[0]['settings']['imgsz']==512
    assert sent[0]['boundary_opt_in'] is True and sent[0]['boundary_limit']==7
    assert api(page,base)['images']==p['images']
