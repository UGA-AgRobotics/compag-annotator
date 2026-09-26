import test from 'node:test';
import assert from 'node:assert/strict';
import { removeLastAssistPoint } from '../../src/compag_annotator/web/assets/assist-review.js';
import { AnnotationCanvas } from '../../src/compag_annotator/web/assets/canvas.js';
import { Editor } from '../../src/compag_annotator/web/assets/editor.js';

test('removal keeps points and labels paired, preserves boxes/masks, invalidates previews without loading geometry', () => {
  let paints=0, inspections=0;
  const controller=new AbortController(), box=[0,0,90,90];
  const e=Object.assign(Object.create(Editor.prototype),{
    alive:true, promptState:{points:[[10,10],[20,20],[30,30]],labels:[1,0,1],box,mode:'single'},
    candidates:[{id:'preview'}],batchJobId:'previous-job',batchPreview:true,assistSignature:'old',
    assistController:controller,
    layer:{previews:[{}],preview:{},requestRender(){paints++;},
      get objects(){throw Error('Do not scan saved masks');},
      get maskCache(){throw Error('Do not load mask geometry');}},
    renderInspector(){inspections++;},
  });
  removeLastAssistPoint(e);
  assert.deepEqual(e.promptState.points,[[10,10],[20,20]]);
  assert.deepEqual(e.promptState.labels,[1,0]);
  assert.equal(e.promptState.box,box);
  assert.equal(e.layer.points,e.promptState.points);
  assert.equal(e.layer.labels,e.promptState.labels);
  assert.equal(controller.signal.aborted,true);
  assert.equal(e.assistSignature,null);
  assert.equal(e.batchJobId,null);
  assert.deepEqual(e.candidates,[]);
  assert.deepEqual(e.layer.previews,[]);
  assert.equal(e.layer.preview,null);
  removeLastAssistPoint(e);removeLastAssistPoint(e);
  assert.deepEqual(e.promptState.points,[]);assert.deepEqual(e.promptState.labels,[]);
  assert.equal(e.promptState.box,box);
  removeLastAssistPoint(e);
  assert.equal(paints,3);assert.equal(inspections,3);
});

test('closed, saving or failed editors never remove prompts', () => {
  for(const state of [{alive:false},{busy:true},{failed:{message:'save failed'}}]){
    const e={alive:true,promptState:{points:[[1,1]],labels:[1]},...state};
    removeLastAssistPoint(e);
    assert.deepEqual(e.promptState.points,[[1,1]]);assert.deepEqual(e.promptState.labels,[1]);
  }
});

function canvas(tool,other={}) {
  const calls={removed:0,added:0,prevented:0};
  const layer=Object.assign(Object.create(AnnotationCanvas.prototype),{
    image:{},canvas:{focus(){},setPointerCapture(){}},tool,space:false,locked:false,
    transform:{x:0,y:0,scale:1},original:()=>[10,10],screen:()=>[20,20],inside:()=>true,
    callbacks:{removeLastPrompt(){calls.removed++;},prompt(){calls.added++;}},...other,
  });
  const click=button=>layer.down({button,pointerId:1,preventDefault(){calls.prevented++;}});
  return {layer,calls,click};
}

test('right-button drag pans in either SAM point tool without deleting or adding points', async () => {
  for(const tool of ['positive','negative']){
    const {layer,calls,click}=canvas(tool);
    await click(2);
    assert.deepEqual(calls,{removed:0,added:0,prevented:0});
    assert.equal(layer.gesture.kind,'pan');
  }
  const {layer,calls,click}=canvas('positive',{locked:true});
  await click(2);assert.equal(calls.removed,0);assert.equal(layer.gesture.kind,'pan');
});

test('ordinary point placement, other tools right-drag, Space and middle-button pan remain intact', async () => {
  const regular=canvas('positive');await regular.click(0);assert.equal(regular.calls.added,1);
  for(const [tool,button,extra] of [
    ['select',2,{}],['pan',2,{}],['prompt-box',2,{}],['brush-add',2,{}],
    ['positive',2,{space:true}],['positive',1,{}],
  ]){
    const {layer,calls,click}=canvas(tool,extra);await click(button);
    assert.equal(layer.gesture.kind,'pan');assert.equal(calls.removed,0);assert.equal(calls.added,0);
  }
});

function keyboardEditor() {
  return Object.assign(Object.create(Editor.prototype),{
    alive:true,promptState:{points:[[10,10],[20,20]],labels:[1,0]},
    layer:{tool:'positive',draft:[],requestRender(){}},
    project:{classes:[]},invalidateAssist(){},syncPrompts(){},renderInspector(){},
    setTool(tool){this.layer.tool=tool;},
  });
}
function keyEvent(changes={}) {
  return {key:'d',code:'KeyD',ctrlKey:true,altKey:false,shiftKey:false,prevented:false,
    target:{closest(){return null;}},preventDefault(){this.prevented=true;},...changes};
}
test('Ctrl+D removes only the last SAM point and prevents bookmarks without activating the D brush', () => {
  const editor=keyboardEditor(),event=keyEvent();
  editor.key(event);
  assert.deepEqual(editor.promptState.points,[[10,10]]);assert.deepEqual(editor.promptState.labels,[1]);
  assert.equal(event.prevented,true);assert.equal(editor.layer.tool,'positive');
  editor.key(keyEvent());editor.key(keyEvent());
  assert.deepEqual(editor.promptState.points,[]);assert.equal(editor.layer.tool,'positive');
  editor.key(keyEvent({ctrlKey:false}));
  assert.equal(editor.layer.tool,'brush-add');
});
test('Ctrl+D respects form, dialog, editable-text and saving guards', () => {
  for(const target of [{closest(){return {};}},
    {closest(){return null;},isContentEditable:true}]){
    const editor=keyboardEditor(),event=keyEvent({target});
    editor.key(event);
    assert.equal(editor.promptState.points.length,2);assert.equal(event.prevented,false);
  }
  const editor=keyboardEditor();editor.busy=true;
  const event=keyEvent();editor.key(event);
  assert.equal(editor.promptState.points.length,2);assert.equal(event.prevented,true);
});
test('physical Ctrl+D works when a different keyboard layout supplies the key text', () => {
  const editor=keyboardEditor(),event=keyEvent({key:'\u0432'});
  editor.key(event);
  assert.equal(editor.promptState.points.length,1);assert.equal(event.prevented,true);
});
