"""Browser evidence uses synthetic images and automated_qa, never human UAT.

These tests run actual project APIs and persistence. Only tests explicitly named
mock_provider use intercepted provider responses; they prove UI protocol only.
"""
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw
from playwright.sync_api import expect


def api(page,path,method="GET",body=None):
    return page.evaluate("""async ([path,method,body]) => {
      const session=await (await fetch('/api/session')).json();
      const response=await fetch(path,{method,headers:{'Content-Type':'application/json','X-Compag-Token':session.token},body:method==='GET'?undefined:JSON.stringify(body)});
      const data=await response.json();if(!response.ok)throw Error(JSON.stringify(data));return data;
    }""",[path,method,body])


def project(page,name="Browser review project",classes=None):
    p=api(page,"/api/projects","POST",{"name":name,"classes":classes if classes is not None else [{"name":"Leaf","color":"#5e965b"},{"name":"Flower","color":"#bc667b"},{"name":"Fruit","color":"#d69b32"}]})
    page.reload();page.get_by_role("heading",name="Projects",exact=True).wait_for()
    page.locator("article").filter(has=page.get_by_role("heading",name=name,exact=True)).get_by_role("button",name="Open workspace").click()
    expect(page.get_by_role("heading",name="Images",exact=True)).to_be_visible()
    return p


def make_image(path,width=1200,height=900):
    image=Image.new("RGB",(width,height),"#eeeade");draw=ImageDraw.Draw(image)
    draw.ellipse((130,120,440,390),fill="#8baa72");draw.polygon([(620,170),(980,230),(830,570)],fill="#dcba70")
    draw.rectangle((190,580,480,770),fill="#a9baca");image.save(path)
    return path


def upload(page,path):
    page.get_by_role("button",name="Import images",exact=True).first.click()
    page.get_by_label("Choose image files",exact=True).set_input_files(str(path))
    page.get_by_role("button",name="Add 1 images to project",exact=True).click()
    expect(page.get_by_text("Import summary",exact=True)).to_be_visible(timeout=20000)
    page.get_by_role("dialog").get_by_role("button",name="Close",exact=True).click()
    expect(page.get_by_role("button",name=f"Annotate {path.name}",exact=True)).to_be_visible()


def scene(page,pid,iid):return api(page,f"/api/projects/{pid}/images/{iid}/annotations")


def wait_objects(page,pid,iid,count):
    until=time.monotonic()+10
    while time.monotonic()<until:
        state=scene(page,pid,iid)
        if len(state["annotations"])==count and page.get_by_text("All changes saved",exact=True).is_visible():return state
        page.wait_for_timeout(60)
    raise AssertionError(f"Expected {count} annotations, got {state}")


def point(page,x,y,width=1200,height=900):
    box=page.get_by_label("Full image annotation canvas",exact=True).bounding_box()
    scale=min((box["width"]-50)/width,(box["height"]-50)/height)
    return (box["x"]+(box["width"]-width*scale)/2+x*scale,box["y"]+(box["height"]-height*scale)/2+y*scale)


def drag(page,a,b):
    page.mouse.move(*a);page.mouse.down();page.mouse.move(*b,steps=8);page.mouse.up()


def test_new_project_class_wizard_and_html_safety(page):
    page.get_by_role("button",name="New project",exact=True).first.click()
    page.get_by_label("Project name",exact=True).fill("<img onerror=alert(1)> Garden ✓")
    page.get_by_role("button",name="Add class",exact=True).click()
    page.get_by_label("Class name",exact=True).fill("Flower ✓ <script>")
    page.get_by_role("button",name="Create project",exact=True).click()
    expect(page.get_by_role("heading",name="Images",exact=True)).to_be_visible()
    assert page.locator(".project-title").inner_text()=="<img onerror=alert(1)> Garden ✓"
    assert page.locator(".project-title img").count()==0
    p=api(page,"/api/projects")[-1]
    state=api(page,f"/api/projects/{p['id']}")
    assert state["classes"][0]["name"]=="Flower ✓ <script>"


def test_full_image_manual_geometry_history_and_review(page,tmp_path):
    p=project(page);path=make_image(tmp_path/"full image ü.png");upload(page,path)
    p=api(page,f"/api/projects/{p['id']}");image=p["images"][0];iid=image["id"]
    page.get_by_role("button",name=f"Annotate {path.name}",exact=True).click()
    expect(page.get_by_text("All changes saved",exact=True)).to_be_visible()
    page.get_by_role("button",name="Fit",exact=True).click()
    page.get_by_role("button",name="Draw box (B)",exact=True).click()
    drag(page,point(page,140,130),point(page,435,390))
    saved=wait_objects(page,p["id"],iid,1);box=saved["annotations"][0]
    assert box["geometry"]["type"]=="box"
    assert all(abs(a-b)<2 for a,b in zip(box["geometry"]["xyxy"],[140,130,435,390]))
    assert box["human_verified"] is False
    page.get_by_role("button",name="Select object 1",exact=True).click()
    page.get_by_role("button",name="Edit geometry (G)",exact=True).click()
    drag(page,point(page,435,390),point(page,470,420))
    expect(page.get_by_text("All changes saved",exact=True)).to_be_visible()
    page.get_by_role("button",name="Draw polygon (P)",exact=True).click()
    for x,y in [(620,170),(980,230),(830,570)]:page.mouse.click(*point(page,x,y))
    page.keyboard.press("Enter");saved=wait_objects(page,p["id"],iid,2)
    polygon=next(a for a in saved["annotations"] if a["geometry"]["type"]=="polygon")
    page.get_by_role("button",name="Select object 2",exact=True).click()
    page.get_by_role("button",name="Edit geometry (G)",exact=True).click()
    page.mouse.dblclick(*point(page,800,200));page.wait_for_timeout(200)
    assert len(scene(page,p["id"],iid)["annotations"][1]["geometry"]["points"])==4
    page.keyboard.down("Shift");page.mouse.click(*point(page,800,200));page.keyboard.up("Shift");page.wait_for_timeout(200)
    assert len(scene(page,p["id"],iid)["annotations"][1]["geometry"]["points"])==3
    page.get_by_role("button",name="Undo",exact=True).click();page.wait_for_timeout(200)
    assert len(scene(page,p["id"],iid)["annotations"][1]["geometry"]["points"])==4
    page.get_by_role("button",name="Redo",exact=True).click();page.wait_for_timeout(200)
    page.get_by_role("button",name="Accept object",exact=True).click();page.wait_for_timeout(200)
    page.get_by_role("button",name="Select object 1",exact=True).click()
    page.get_by_role("button",name="Accept object",exact=True).click();page.wait_for_timeout(200)
    page.get_by_role("button",name="Mark reviewed",exact=True).click()
    page.get_by_role("checkbox",name="I reviewed all objects",exact=False).check()
    page.get_by_role("button",name="Confirm full image review",exact=True).click()
    expect(page.get_by_role("dialog")).not_to_be_visible()
    p=api(page,f"/api/projects/{p['id']}")
    assert p["images"][0]["complete"] is True
    assert p["images"][0]["review_actor"]=="automated_qa"
    assert not any(a["human_verified"] for a in scene(page,p["id"],iid)["annotations"])
    page.screenshot(path=str(tmp_path/"editor.png"),full_page=True)
    page.set_viewport_size({"width":1366,"height":768})
    page.get_by_role("button",name="Fit",exact=True).click()
    canvas_bounds=page.get_by_label("Full image annotation canvas",exact=True).bounding_box()
    assert canvas_bounds["width"]>500 and canvas_bounds["height"]>450
    assert page.evaluate("document.documentElement.scrollWidth")<=1366
    page.screenshot(path=str(tmp_path/"editor-laptop.png"),full_page=True)
    page.reload();page.locator("article").filter(has=page.get_by_role("heading",name="Browser review project",exact=True)).get_by_role("button",name="Open workspace").click()
    page.get_by_role("button",name=f"Annotate {path.name}",exact=True).click();wait_objects(page,p["id"],iid,2)


def test_all_navigation_models_training_and_offline_help(page,tmp_path):
    project(page,"Navigation project")
    for nav,heading in [("Labels","Labels"),("Models & AI","Models & AI"),("Rounds","Review rounds"),("Train","Train"),("Export","Export annotations"),("Jobs","Jobs"),("Help & Settings","Help & settings")]:
        page.get_by_role("navigation").get_by_role("button",name=nav,exact=True).click()
        expect(page.get_by_role("heading",name=heading,exact=True)).to_be_visible()
        assert page.get_by_text("This view could not load",exact=True).count()==0
        if nav=="Train":
            page.get_by_text("Advanced settings",exact=True).click()
            assert not page.get_by_role("checkbox",name="Check/repair tile-edge masks before training (slower)").is_checked()
        page.screenshot(path=str(tmp_path/(nav.replace(" & ","-").replace(" ","-").lower()+".png")),full_page=True)
    expect(page.get_by_role("heading",name="Help & settings",exact=True)).to_be_visible()
    assert api(page,"/api/doctor")["boundary_invocations"]==0


def test_exact_mask_brush_holes_components_and_zoom_pan(page,tmp_path):
    p=project(page,"Mask brush project");path=make_image(tmp_path/"mask scene.png");upload(page,path)
    p=api(page,f"/api/projects/{p['id']}");iid=p["images"][0]["id"]
    page.get_by_role("button",name=f"Annotate {path.name}",exact=True).click()
    expect(page.get_by_text("All changes saved",exact=True)).to_be_visible()
    page.get_by_role("button",name="Fit",exact=True).click()
    page.get_by_role("button",name="Add mask pixels (D)",exact=True).click()
    page.get_by_label("Brush radius",exact=True).fill("80")
    page.mouse.click(*point(page,260,250));wait_objects(page,p["id"],iid,1)
    page.mouse.click(*point(page,510,250));wait_objects(page,p["id"],iid,1)
    page.get_by_role("button",name="Subtract mask pixels (E)",exact=True).click()
    page.get_by_label("Brush radius",exact=True).fill("25")
    page.mouse.click(*point(page,260,250));wait_objects(page,p["id"],iid,1)
    obj=scene(page,p["id"],iid)["annotations"][0]
    geometry=api(page,obj["geometry_url"])
    assert geometry["type"]=="mask"
    pixels=page.evaluate("""async g=>{const {decodeRLE}=await import('/assets/geometry.js');const m=decodeRLE(g.rle);return [[260,250],[310,250],[510,250],[400,250]].map(([x,y])=>m.pixels[y*m.width+x]);}""",geometry)
    assert pixels==[0,1,1,0],"Hole and disconnected component must remain exact"
    page.get_by_role("checkbox",name="Show object 1",exact=True).uncheck()
    page.get_by_role("checkbox",name="Show object 1",exact=True).check()
    original_bounds=page.get_by_label("Full image annotation canvas",exact=True).bounding_box()
    before_zoom=int(page.locator(".canvas-hud>span").inner_text().rstrip("%"))
    page.get_by_role("button",name="Zoom in",exact=True).click()
    after_zoom=int(page.locator(".canvas-hud>span").inner_text().rstrip("%"))
    assert after_zoom>before_zoom
    page.keyboard.down("Space");drag(page,point(page,600,600),point(page,650,650));page.keyboard.up("Space")
    assert page.get_by_label("Full image annotation canvas",exact=True).bounding_box()==original_bounds
    page.get_by_role("button",name="Fit",exact=True).click()
    page.get_by_role("button",name="Undo",exact=True).click();page.wait_for_timeout(150)
    geometry=api(page,obj["geometry_url"])
    assert page.evaluate("""async g=>{const {decodeRLE}=await import('/assets/geometry.js');const m=decodeRLE(g.rle);return m.pixels[250*m.width+260];}""",geometry)==1
    page.get_by_role("button",name="Redo",exact=True).click();page.wait_for_timeout(150)
    page.screenshot(path=str(tmp_path/"exact-mask-editor.png"),full_page=True)


def test_folder_import_filename_round_order_class_mapping_and_export(page,tmp_path):
    p=project(page,"Folder and rounds project")
    folder=tmp_path/"images with spaces";folder.mkdir()
    make_image(folder/"first.png",1200,900);make_image(folder/"second.png",1201,900);make_image(folder/"third.png",1202,900)
    page.get_by_role("button",name="Import images",exact=True).first.click()
    page.get_by_label("Choose a folder",exact=True).set_input_files(str(folder))
    page.get_by_role("button",name="Add 3 images to project",exact=True).click()
    expect(page.get_by_text("Import summary",exact=True)).to_be_visible(timeout=20000)
    page.get_by_role("dialog").get_by_role("button",name="Close",exact=True).click()
    p=api(page,f"/api/projects/{p['id']}");initial_order=[i["id"] for i in p["images"]]
    first_name=p["images"][0]["name"]
    page.get_by_role("navigation").get_by_role("button",name="Rounds",exact=True).click()
    page.get_by_label("Number of rounds",exact=True).fill("2")
    page.get_by_label("Order",exact=True).select_option("manual")
    page.get_by_role("button",name=f"Move {first_name} down",exact=True).click()
    page.get_by_role("button",name="Preview distribution",exact=True).click()
    expect(page.get_by_text("Preview image allocation",exact=True)).to_be_visible()
    page.get_by_role("button",name="Confirm round plan",exact=True).click()
    expect(page.get_by_role("heading",name="Round 1",exact=True)).to_be_visible()
    p=api(page,f"/api/projects/{p['id']}");assert [len(r["image_ids"]) for r in p["rounds"]]==[2,1]
    assert p["rounds"][0]["image_ids"][0]==initial_order[1]
    page.get_by_role("button",name="Start round",exact=True).first.click()
    expect(page.get_by_role("button",name="Finish review",exact=True)).to_be_visible()
    page.get_by_role("navigation").get_by_role("button",name="Export",exact=True).click()
    coco={"images":[{"id":1,"file_name":"first.png","width":1200,"height":900}],"categories":[{"id":7,"name":"Imported leaf"}],"annotations":[{"id":1,"image_id":1,"category_id":7,"segmentation":[[10,10,80,10,80,90,10,90]],"bbox":[10,10,70,80],"area":5600,"iscrowd":0}]}
    source=tmp_path/"annotation.json";source.write_text(json.dumps(coco))
    page.get_by_text("Import existing annotations",exact=True).click()
    page.get_by_label("Annotation format",exact=True).select_option("coco")
    page.get_by_label("Annotation file / archive",exact=True).set_input_files(str(source))
    page.get_by_label("Project class for Imported leaf",exact=True).select_option(p["classes"][0]["id"])
    page.get_by_role("button",name="Import annotations",exact=True).click()
    expect(page.get_by_role("heading",name="Jobs",exact=True)).to_be_visible()
    until=time.monotonic()+10
    while time.monotonic()<until:
        jobs=api(page,"/api/jobs");job=next((j for j in jobs if j["kind"]=="import_annotations" and j["payload"]["project_id"]==p["id"]),None)
        if job and job["status"] in ("complete","failed"):break
        page.wait_for_timeout(100)
    assert job["status"]=="complete",job
    image=next(i for i in p["images"] if i["name"]=="first.png")
    obj=scene(page,p["id"],image["id"])["annotations"][0]
    assert obj["class_id"]==p["classes"][0]["id"] and obj["human_verified"] is False
    page.get_by_role("navigation").get_by_role("button",name="Export",exact=True).click()
    page.get_by_text("Project backup — continue editing later",exact=True).click()
    page.get_by_role("button",name="Create project backup",exact=True).click()
    expect(page.get_by_role("link",name="Download artifact",exact=True)).to_be_visible(timeout=15000)
    assert api(page,"/api/doctor")["boundary_invocations"]==0


def test_mock_provider_sam_prompt_binding_alternatives_and_stale_rejection(page,tmp_path):
    """Protocol-only provider interception: not evidence of real SAM inference."""
    p=project(page,"SAM protocol test");path=make_image(tmp_path/"sam input.png");upload(page,path)
    p=api(page,f"/api/projects/{p['id']}");im=p["images"][0];iid=im["id"]
    provider={"id":"mock-sam-protocol","provider":"sam2","name":"Mock SAM protocol fixture","architecture":"mock-only"}
    page.route("**/api/models",lambda route:route.fulfill(json=[provider]))
    requests=[];mode={"stale":True}
    def assist(route):
        body=route.request.post_data_json;requests.append(body)
        route.fulfill(json={"id":"mock-preview-job","kind":"assist","status":"queued","progress":{"stage":"Mock protocol response"}})
    def result(route):
        body=requests[-1]
        binding={"image_id":iid,"image_sha256":im["sha256"],"project_revision":body["expected_revision"],"annotation_id":body.get("annotation_id"),"model_id":body["model_id"],"class_id":body["class_id"],"points":body["points"],"labels":body["labels"],"box":body.get("box")}
        if mode["stale"]:binding["project_revision"]-=1
        # Exact full-resolution masks in uncompressed COCO column-major order.
        from compag_annotator.geometry import encode_rle
        import numpy as np
        mask=np.zeros((900,1200),dtype=bool);mask[150:300,160:350]=True
        alternative=np.zeros_like(mask);alternative[145:310,155:360]=True
        route.fulfill(json={"id":"mock-preview-job","kind":"assist","status":"complete","progress":{},"result":{"provider":"sam2","model_id":provider["id"],"binding":binding,"annotations":[{"geometry":{"type":"mask","rle":encode_rle(mask)},"score":.8}],"alternatives":[{"geometry":{"type":"mask","rle":encode_rle(alternative)},"score":.7}]}})
    page.route(f"**/images/{iid}/assist",assist);page.route("**/api/jobs/mock-preview-job",result)
    page.get_by_role("button",name=f"Annotate {path.name}",exact=True).click()
    page.get_by_label('Prompt mode',exact=True).select_option('single')
    page.get_by_label("Model",exact=True).select_option(provider["id"])
    page.get_by_role("button",name="Fit",exact=True).click()
    page.get_by_role("button",name="SAM positive point",exact=True).click();page.mouse.click(*point(page,220,220))
    page.get_by_role("button",name="SAM negative point",exact=True).click();page.mouse.click(*point(page,410,330))
    page.get_by_role("button",name="SAM box prompt",exact=True).click();drag(page,point(page,140,130),point(page,400,350))
    page.get_by_role("button",name="Preview mask",exact=True).click()
    expect(page.get_by_text("This preview belongs to an earlier image, edit, model or prompt. Request a new preview.",exact=True).first).to_be_visible()
    assert page.get_by_role("button",name="Use preview as draft",exact=True).count()==0
    assert requests[0]["labels"]==[1,0] and len(requests[0]["box"])==4
    assert requests[0]["class_id"]==p["classes"][0]["id"]
    assert not scene(page,p["id"],iid)["annotations"]
    mode["stale"]=False
    page.get_by_role("button",name="Preview mask",exact=True).click()
    expect(page.get_by_role("button",name="Use preview as draft",exact=True)).to_be_visible()
    page.get_by_label("Alternative preview",exact=True).select_option("1")
    page.get_by_role("button",name="Use preview as draft",exact=True).click()
    saved=wait_objects(page,p["id"],iid,1)["annotations"][0]
    assert saved["source"]["kind"]=="sam2" and saved["source"]["model_id"]==provider["id"]
    assert saved["status"]=="draft" and saved["human_verified"] is False


def test_revision_conflict_is_visible_and_keeps_unsaved_geometry(page,tmp_path):
    p=project(page,"Revision conflict project");path=make_image(tmp_path/"conflict image.png");upload(page,path)
    p=api(page,f"/api/projects/{p['id']}");iid=p["images"][0]["id"]
    page.get_by_role("button",name=f"Annotate {path.name}",exact=True).click()
    expect(page.get_by_text("All changes saved",exact=True)).to_be_visible()
    api(page,f"/api/projects/{p['id']}","PATCH",{"name":"Changed in another client","expected_revision":p["revision"]})
    page.get_by_role("button",name="Draw box (B)",exact=True).click()
    drag(page,point(page,120,100),point(page,250,240))
    expect(page.get_by_text("Changes not saved",exact=True)).to_be_visible()
    assert not scene(page,p["id"],iid)["annotations"]
    page.get_by_role("button",name="Reload saved state",exact=True).click()
    page.get_by_role("dialog").get_by_role("button",name="Reload",exact=True).click()
    expect(page.get_by_text("All changes saved",exact=True)).to_be_visible()
    page.get_by_role("button",name="Draw box (B)",exact=True).click()
    drag(page,point(page,120,100),point(page,250,240));wait_objects(page,p["id"],iid,1)


def test_mock_provider_setup_train_and_predict_form_contracts(page,tmp_path):
    """Intercept provider submissions only; no installs, downloads or GPU jobs."""
    p=project(page,"Provider form contract");path=make_image(tmp_path/"provider form image.png");upload(page,path)
    upload(page,make_image(tmp_path/"separate validation.png",width=1201))
    p=api(page,f"/api/projects/{p['id']}")
    for index,image in enumerate(p['images']):
        p=api(page,f"/api/projects/{p['id']}/images/{image['id']}/annotations","POST",{
            "geometry":{"type":"polygon","points":[[10,10],[40,10],[40,40]]},
            "class_id":p['classes'][0]['id'],"status":"accepted","expected_revision":p['revision']})
        p=api(page,f"/api/projects/{p['id']}/images/{image['id']}","PATCH",{
            "role":"validation" if index else "pool","complete":True,"attest":True,"expected_revision":p['revision']})
    models=[{"id":"mock-yolo","name":"Mock YOLO form fixture","provider":"yolo","class_names":{"0":"Leaf"}}, {"id":"mock-sam","name":"Mock SAM form fixture","provider":"sam2"}]
    page.route("**/api/models",lambda route:route.fulfill(json=models))
    submitted={}
    def capture(kind):
        def handle(route):
            submitted[kind]=route.request.post_data_json
            route.fulfill(json={"id":f"mock-{kind}","kind":kind,"status":"cancelled","progress":{"stage":"UI protocol test; no provider ran"}})
        return handle
    page.route("**/api/models/install",capture("install_model"))
    page.route(f"**/api/projects/{p['id']}/train",capture("train"))
    page.route(f"**/api/projects/{p['id']}/predict",capture("predict"))
    page.get_by_role("navigation").get_by_role("button",name="Models & AI",exact=True).click()
    page.get_by_role("button",name="Install runtime",exact=True).click()
    assert not page.get_by_role("checkbox",name="Download this selected checkpoint",exact=True).is_checked()
    assert page.get_by_role("checkbox",name="Download this selected checkpoint",exact=True).is_disabled()
    page.get_by_role("button",name="Start installation",exact=True).click()
    assert "install_model" not in submitted
    page.get_by_role("checkbox",name="I reviewed the source and applicable license terms and approve this installation.",exact=True).check()
    page.get_by_role("button",name="Start installation",exact=True).click()
    expect(page.get_by_role("heading",name="Jobs",exact=True)).to_be_visible()
    assert submitted["install_model"]["provider"]=="sam3" and submitted["install_model"]["download_weights"] is False
    page.get_by_role("navigation").get_by_role("button",name="Train",exact=True).click()
    page.get_by_label("Base / previous segmentation model",exact=True).select_option("mock-yolo")
    expect(page.get_by_label("Device",exact=True)).to_be_enabled()
    page.get_by_role("button",name="Train model",exact=True).click()
    expect(page.get_by_role("heading",name="Jobs",exact=True)).to_be_visible()
    assert submitted["train"]["boundary_opt_in"] is False
    assert submitted["train"]["allow_lossy"] is False and submitted["train"]["auto_activate"] is False
    assert submitted["train"]["qa_smoke"] is True
    page.get_by_role("navigation").get_by_role("button",name="Train",exact=True).click()
    page.get_by_label("Base / previous segmentation model",exact=True).select_option("mock-yolo")
    page.get_by_text("Advanced settings",exact=True).click()
    page.get_by_role("checkbox",name="Check/repair tile-edge masks before training (slower)",exact=True).check()
    page.get_by_label("Boundary preparation SAM model",exact=True).select_option("mock-sam")
    expect(page.get_by_label("Device",exact=True)).to_be_enabled()
    page.get_by_role("button",name="Train model",exact=True).click()
    expect(page.get_by_role("heading",name="Jobs",exact=True)).to_be_visible()
    assert submitted["train"]["boundary_opt_in"] is True and submitted["train"]["boundary_model_id"]=="mock-sam"
    page.get_by_role("navigation").get_by_role("button",name="Images",exact=True).click()
    page.get_by_role("checkbox",name=f"Select {path.name}",exact=True).check()
    page.get_by_role("button",name="Predict selected",exact=True).click()
    expect(page.get_by_role("dialog").get_by_label("Device",exact=True)).to_be_enabled()
    page.get_by_role("button",name="Create proposals",exact=True).click()
    expect(page.get_by_role("heading",name="Jobs",exact=True)).to_be_visible()
    assert submitted["predict"]["settings"]["confidence"]==.25
    assert submitted["predict"]["settings"]["tiling"]=={"mode":"tiled","preset":"512","width":512,"height":512,"overlap":{"mode":"pixels","x":0,"y":0}}
    assert submitted["predict"]["boundary_opt_in"] is False
    assert submitted["train"]["training_mode"]=="tiles_512" and submitted["train"]["imgsz"]==512
    assert submitted["predict"]["image_ids"]==[p["images"][0]["id"]]
    assert api(page,"/api/doctor")["boundary_invocations"]==0


def test_staged_boundary_comparison_and_review_without_provider(page,tmp_path):
    """A staged synthetic proposal exercises live review API; no SAM is run."""
    from compag_annotator.core.projects import Project
    p=project(page,"Boundary review fixture");path=make_image(tmp_path/"boundary fixture.png");upload(page,path)
    p=api(page,f"/api/projects/{p['id']}");iid=p["images"][0]["id"]
    before={"type":"polygon","points":[[130,120],[400,120],[400,390],[130,390]]}
    after={"type":"polygon","points":[[120,110],[440,110],[440,400],[120,400]]}
    p=api(page,f"/api/projects/{p['id']}/images/{iid}/annotations","POST",{"geometry":before,"class_id":p["classes"][0]["id"],"status":"accepted","expected_revision":p["revision"]})
    obj=scene(page,p["id"],iid)["annotations"][0]
    local=Project(p["path"])
    with local.edit(p["revision"],"browser_test_staged_fixture","automated_qa") as (_,state):
        state["boundary_proposals"].append({"id":"synthetic-stage","annotation_id":obj["id"],"annotation_revision":obj["revision"],"image_id":iid,"class_id":obj["class_id"],"model_id":"mock-boundary-source","status":"pending","geometry":after,"reason":"Synthetic browser comparison fixture; not provider inference","human_verified":False})
    page.get_by_role("navigation").get_by_role("button",name="Train",exact=True).click()
    page.get_by_role("button",name="Compare & review",exact=True).click()
    expect(page.get_by_label("Current boundary geometry",exact=True)).to_be_visible()
    expect(page.get_by_label("Proposed boundary geometry",exact=True)).to_be_visible()
    expect(page.get_by_text("Geometry comparison ready. Inspect both masks before choosing a decision.",exact=True)).to_be_visible()
    page.screenshot(path=str(tmp_path/"boundary-comparison.png"),full_page=True)
    page.get_by_role("button",name="Retain original",exact=True).click()
    expect(page.get_by_role("dialog")).not_to_be_visible()
    props=api(page,f"/api/projects/{p['id']}/boundary/proposals")
    assert props[0]["status"]=="retained" and props[0]["review_actor"]=="automated_qa"
    assert api(page,obj["geometry_url"])==before
    assert api(page,"/api/doctor")["boundary_invocations"]==0
