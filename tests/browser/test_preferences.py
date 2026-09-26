"""UI integration with explicit device/API doubles; no real training or download."""
from playwright.sync_api import expect
from test_app import project, make_image, upload, api
from test_addendum import generate


def device_report(route):
    route.fulfill(json={'devices':[{'id':'cpu','name':'CPU','kind':'cpu'},{'id':'cuda:0','name':'Test GPU','kind':'gpu'}], 'cuda_available':True, 'message':'Explicit UI test double'})


def test_dark_theme_persists_without_changing_image_styles(page):
    page.get_by_role('combobox',name='Theme',exact=True).select_option('dark')
    assert page.locator('html').get_attribute('data-theme') == 'dark'
    page.reload(); expect(page.get_by_role('combobox',name='Theme',exact=True)).to_have_value('dark')
    page.get_by_role('button',name='New project',exact=True).first.click()
    assert page.get_by_role('dialog').evaluate('(d)=>getComputedStyle(d).backgroundColor') == 'rgb(23, 29, 25)'
    assert page.locator('img').first.evaluate('(d)=>getComputedStyle(d).filter') == 'none'
    page.get_by_role('dialog').get_by_role('button',name='Close',exact=True).click()
    page.get_by_role('combobox',name='Theme',exact=True).select_option('light')
    assert page.locator('html').get_attribute('data-theme') == 'light'


def test_generation_device_and_paper_settings_payload(mock_provider_page,tmp_path):
    page=mock_provider_page
    page.route('**/api/models/devices?**',device_report)
    p=project(page,'Device and paper settings QA',classes=[])
    path=make_image(tmp_path/'device-test.png');upload(page,path)
    captured=[]
    def capture(route):
        captured.append(route.request.post_data_json)
        route.fulfill(status=400,json={'detail':'Explicit submission interception; no generation'})
    page.route('**/api/projects/*/generate',capture)
    page.get_by_role('button',name='Generate masks',exact=True).click()
    d=page.get_by_role('dialog',name='Generate masks',exact=True)
    expect(d.get_by_label('Sampling density',exact=True)).to_have_value('64')
    d.get_by_text('Advanced automatic mask settings',exact=True).click()
    expect(d.get_by_label('Points per batch',exact=True)).to_have_value('512')
    expect(d.get_by_label('Stability threshold',exact=True)).to_have_value('0.88')
    d.get_by_label('Device',exact=True).select_option('cuda:0')
    d.get_by_label('Mask logit threshold',exact=True).fill('1.2')
    d.get_by_label('Stability score offset',exact=True).fill('1.5')
    d.get_by_role('checkbox',name='Generate alternative masks per point',exact=True).uncheck()
    d.get_by_role('button',name='Generate masks',exact=True).click()
    expect(page.get_by_text('Explicit submission interception; no generation',exact=True)).to_be_visible()
    assert captured[0]['device']=='cuda:0'
    assert captured[0]['settings']['mask_threshold']==1.2
    assert captured[0]['settings']['stability_score_offset']==1.5
    assert captured[0]['settings']['multimask_output'] is False
    assert not captured[0]['boundary_opt_in']


def test_training_selects_yolo_runtime_device_and_submits(page,tmp_path):
    calls=[]
    def report(route): calls.append(route.request.url);device_report(route)
    page.route('**/api/models/devices?**',report)
    page.route('**/api/models',lambda r:r.fulfill(json=[{'id':'yolo-test','provider':'yolo','name':'Explicit mock YOLO','architecture':'custom-seg'}]))
    p=project(page,'Training device UI QA')
    for index in range(2):upload(page,make_image(tmp_path/f'train-device-{index}.png',width=1200+index))
    p=api(page,f"/api/projects/{p['id']}")
    for index,image in enumerate(p['images']):
        p=api(page,f"/api/projects/{p['id']}/images/{image['id']}/annotations","POST",{
            'geometry':{'type':'polygon','points':[[10,10],[40,10],[40,40]]},
            'class_id':p['classes'][0]['id'],'status':'accepted','expected_revision':p['revision']})
        p=api(page,f"/api/projects/{p['id']}/images/{image['id']}","PATCH",{
            'role':'validation' if index else 'pool','complete':True,'attest':True,'expected_revision':p['revision']})
    captured=[]
    def capture(route):
        captured.append(route.request.post_data_json)
        route.fulfill(status=400,json={'detail':'Training intercepted; no ML execution'})
    page.route('**/api/projects/*/train',capture)
    page.get_by_role('navigation').get_by_role('button',name='Train',exact=True).click()
    page.get_by_label('Device',exact=True).select_option('cuda:0')
    page.get_by_label('Base / previous segmentation model',exact=True).select_option('yolo-test')
    page.get_by_role('button',name='Train model',exact=True).click()
    expect(page.get_by_text('Training intercepted; no ML execution',exact=True)).to_be_visible()
    assert captured[0]['device']=='cuda:0' and all('provider=yolo' in c for c in calls)


def test_device_picker_unavailable_gpu_and_provider_race(page):
    result=page.evaluate('''async () => {
      const {DevicePicker}=await import('/assets/devices.js');
      const resolvers={};const app={device:'cuda:1',api:{get:p=>new Promise(resolve=>resolvers[p.includes('sam3')?'sam3':'sam2']=resolve)}};
      const picker=new DevicePicker(app,'sam2');document.body.append(picker.root);
      const pending=picker.load('sam3');
      resolvers.sam3({devices:[{id:'cpu',kind:'cpu'},{id:'cuda:0',kind:'gpu',name:'Test GPU'}]});await pending;
      resolvers.sam2({devices:[{id:'cpu',kind:'cpu'},{id:'cuda:1',kind:'gpu',name:'Stale GPU'}]});await new Promise(r=>setTimeout(r,0));
      let rejected=false;try{picker.value()}catch(e){rejected=true}
      return {rejected,selected:picker.control.value,available:[...picker.available],cpuDisabled:picker.control.querySelector('[value="cpu"]').disabled};
    }''')
    assert result == {'rejected':True,'selected':'cuda:1','available':['cuda:0'],'cpuDisabled':True}


def test_progress_uses_real_bytes_and_setup_step_basis(page):
    values=page.evaluate('''async () => {
      const {jobCard}=await import('/assets/jobs.js');
      return [{bytes:25,total_bytes:100,stage:'download'}, {bytes:25,total_bytes:null,stage:'download'},
        {setup_percent:25,setup_total:8,setup_completed:2,download_bytes:50,download_total_bytes:100,download_label:'test.whl'}].map(progress => {
          const card=jobCard({}, {id:'test',kind:'install_model',status:'running',progress});
          return {text:card.textContent,bar:card.querySelector('progress').getAttribute('value')};
        });
    }''')
    assert '25.0%' in values[0]['text'] and values[0]['bar']=='0.25'
    assert 'total size unavailable' in values[1]['text'] and values[1]['bar'] is None
    assert '25.0%' in values[2]['text'] and '2/8 setup steps' in values[2]['text'] and '50.0%' in values[2]['text']


def test_sam_prompt_settings_reach_request(mock_provider_page,tmp_path):
    page=mock_provider_page;page.route('**/api/models/devices?**',device_report)
    p=project(page,'SAM prompt controls QA');path=make_image(tmp_path/'prompt-control.png');upload(page,path)
    page.get_by_role('button',name=f'Annotate {path.name}',exact=True).click()
    page.get_by_label('Prompt mode',exact=True).select_option('single')
    page.get_by_label('Model',exact=True).select_option('automated-qa-sam2')
    page.get_by_role('button',name='SAM settings',exact=True).click()
    d=page.get_by_role('dialog',name='SAM settings',exact=True)
    d.get_by_label('Device',exact=True).select_option('cuda:0')
    d.get_by_label('Mask logit threshold',exact=True).fill('2')
    d.get_by_role('checkbox',name='Offer alternative masks',exact=True).uncheck()
    d.get_by_label('Precision',exact=True).select_option('bfloat16')
    d.get_by_role('button',name='Apply SAM settings',exact=True).click()
    expect(d).not_to_be_visible()
    page.get_by_role('button',name='SAM positive point',exact=True).click()
    page.get_by_label('Full image annotation canvas',exact=True).click()
    captured=[]
    def capture(route):
        captured.append(route.request.post_data_json)
        route.fulfill(status=400,json={'detail':'Prompt intercepted; no ML execution'})
    page.route('**/api/projects/*/images/*/assist',capture)
    page.get_by_role('button',name='Preview mask',exact=True).click()
    expect(page.get_by_text('Prompt intercepted; no ML execution',exact=True).first).to_be_visible()
    assert captured[0]['device']=='cuda:0'
    assert captured[0]['settings']=={'mask_threshold':2,'multimask_output':False,'precision':'bfloat16'}
