import test from 'node:test';
import assert from 'node:assert/strict';
import { Editor } from '../../src/compag_annotator/web/assets/editor.js';
function editor(mode='independent') {
  const requests=[];let body;
  const row={geometry:{type:'mask',rle:{size:[2,2],counts:[0,4]}},source:{kind:'sam2'}};
  const value=Object.assign(Object.create(Editor.prototype),{
    alive:true,busy:false,failed:null,revision:7,path:'/image',image:{id:'i',sha256:'image'},
    promptState:{mode,points:[[1,1],[2,2]],labels:[1,1],box:null,model_id:'sam',class_id:null},
    layer:{selected:'old-object',select(id){this.selected=id;},render(){},previews:[]},
    app:{device:'cpu',api:{post:async(path,payload)=>{requests.push({path,payload});body=payload;return {id:'job'};},
      waitJob:async()=>({id:'job',status:'complete',result:{mode,requested_points:2,empty_points:[],
        binding:{mode,image_id:'i',image_sha256:'image',project_revision:7,
          annotation_id:body.annotation_id||null,model_id:'sam',class_id:null,
          points:body.points,labels:body.labels,box:null},
        annotations:mode==='independent'?[row,row]:[row],alternatives:[row]}})}},
    renderInspector(){},
  });
  return {value,requests};
}
test('all independent masks are shown together, not as mutually exclusive alternatives',async()=>{
  const {value,requests}=editor();await value.assist();
  assert.equal(value.layer.selected,null);
  assert.equal(requests[0].payload.annotation_id,undefined);
  assert.equal(value.batchPreview,true);assert.equal(value.batchJobId,'job');
  assert.equal(value.candidates.length,2);assert.equal(value.layer.previews.length,2);
  assert.equal(value.layer.preview,null);assert.equal(value.assistController,null);
  assert.match(value.assistMessage,/2 independent masks ready/);
  value.mutate=async fn=>{await fn();return true;};value.syncPrompts=()=>{};
  await value.acceptPreview();
  assert.deepEqual(requests[1],{path:'/image/assist/job/apply',payload:{expected_revision:7}});
  assert.deepEqual(value.promptState.points,[]);
});
test('single-object refinement keeps the target and offers alternatives',async()=>{
  const {value,requests}=editor('single');await value.assist();
  assert.equal(requests[0].payload.annotation_id,'old-object');
  assert.equal(value.batchPreview,false);assert.equal(value.candidates.length,2);
  assert.deepEqual(value.layer.previews,[]);assert.ok(value.layer.preview);
});
test('changing the prompt invalidates every batch preview and its save target',async()=>{
  const {value}=editor();await value.assist();
  const original=value.assistSignature;value.promptState.points.push([3,3]);
  assert.notEqual(value.signature(),original);
  value.invalidateAssist();assert.equal(value.batchJobId,null);
  assert.equal(value.layer.previews.length,0);assert.equal(value.candidates.length,0);
  await assert.rejects(value.acceptPreview());
});
