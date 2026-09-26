// Run with: node --experimental-default-type=module --test tests/js/review_visibility.test.js
// Synthetic overlapping shapes exercise the production selection paths; no model inference.
import test from 'node:test';
import assert from 'node:assert/strict';
import { AnnotationCanvas } from '../../src/compag_annotator/web/assets/canvas.js';
import { activeSelection } from '../../src/compag_annotator/web/assets/selection.js';
import { Editor } from '../../src/compag_annotator/web/assets/editor.js';

function scene() {
  const layer = Object.assign(Object.create(AnnotationCanvas.prototype), {
    objects: [
      {id:'under', class_id:null, status:'proposal', geometry:{type:'polygon', points:[[0,0],[10,0],[10,10],[0,10]]}},
      {id:'top', class_id:'fruit', status:'draft', geometry:{type:'polygon', points:[[0,0],[10,0],[10,10],[0,10]]}},
    ],
    hiddenClasses:new Set(), hiddenObjects:new Set(), hiddenLayers:new Set(),
    hideAssigned:false, showUnassigned:true, showProposals:true,
    showRejected:false, showSuperseded:false, selection:new Set(), selected:null,
    callbacks:{}, render:()=>{}, maskCache:{ensure:async()=>{}},
  });
  return layer;
}

test('assigned overlay hides without changing annotations and the underlying object becomes the hit', async()=>{
  const layer=scene(), before=JSON.stringify(layer.objects);
  assert.deepEqual((await layer.hits([5,5])).map(o=>o.id), ['top','under']);
  layer.select('top'); layer.select('under',true);
  layer.overlaps=['top','under']; layer.hover='top';
  layer.hideAssigned=true; layer.pruneHiddenSelection();
  assert.deepEqual((await layer.hits([5,5])).map(o=>o.id), ['under']);
  assert.deepEqual([...layer.selection],['under']);
  assert.deepEqual(layer.overlaps,['under']);
  assert.equal(layer.hover,null);
  layer.select('top',true);
  assert.deepEqual([...layer.selection],['under']);
  assert.deepEqual(activeSelection({layer,objects:layer.objects}).map(o=>o.id),['under']);
  layer.hideAssigned=false;
  assert.deepEqual((await layer.hits([5,5])).map(o=>o.id),['top','under']);
  assert.equal(JSON.stringify(layer.objects),before);
});

test('class and object controls compose with assigned filter and keep saved geometry intact',()=>{
  const layer=scene(), before=structuredClone(layer.objects);
  layer.select('top');
  layer.hiddenClasses.add('fruit'); layer.pruneHiddenSelection();
  assert.equal(layer.selected,null); assert.equal(layer.selection.size,0);
  assert.equal(layer.visible(layer.objects[0]),true);
  layer.hideAssigned=true; layer.hiddenClasses.clear();
  assert.equal(layer.visible(layer.objects[1]),false);
  layer.hideAssigned=false; layer.hiddenObjects.add('top');
  assert.equal(layer.visible(layer.objects[1]),false);
  layer.hiddenObjects.clear();
  assert.equal(layer.visible(layer.objects[1]),true);
  assert.deepEqual(layer.objects,before);
});

test('reload after assignment, undo and redo follows the current class without losing objects',async()=>{
  const layer=scene(); layer.hideAssigned=true; layer.select('under');
  let rows=structuredClone(layer.objects);
  const editor=Object.assign(Object.create(Editor.prototype),{
    layer,objects:layer.objects,objectOrder:new Map(),alive:true,busy:false,path:'/image',
    app:{api:{get:async()=>({annotations:structuredClone(rows),revision:1})}},
    renderInspector:()=>{},setStatus:()=>{},
  });
  layer.maskCache.clear=()=>{};
  rows[0].class_id='fruit'; rows[0].status='draft';
  await editor.loadScene();
  assert.equal(layer.selected,null); assert.equal(layer.selection.size,0);
  assert.equal(layer.objects.filter(o=>layer.visible(o)).length,0);
  rows[0].class_id=null; rows[0].status='proposal';
  await editor.loadScene();
  assert.deepEqual(layer.objects.filter(o=>layer.visible(o)).map(o=>o.id),['under']);
  rows[0].class_id='fruit';
  await editor.loadScene();
  assert.equal(layer.objects.filter(o=>layer.visible(o)).length,0);
  assert.equal(layer.objects.length,2);
});

test('a display change during asynchronous geometry loading excludes the newly hidden hit',async()=>{
  const layer=scene();
  let release;
  const ready=new Promise(resolve=>{release=resolve;});
  layer.maskCache.ensure=()=>ready;
  const pending=layer.hits([5,5]);
  layer.hideAssigned=true; release();
  assert.deepEqual((await pending).map(o=>o.id),['under']);
});

test('pending review is distinct from assignment and completed decisions',async()=>{
  const {reviewStatus,reviewCounts}=await import('../../src/compag_annotator/web/assets/review-status.js');
  const layer=scene();
  layer.objects.push({id:'accepted',class_id:'fruit',status:'accepted'}, {id:'rejected',class_id:null,status:'rejected'});
  const before=structuredClone(layer.objects);
  assert.equal(reviewStatus(layer.objects[1]),'Labeled · needs review');
  assert.deepEqual(reviewCounts(layer.objects),{pending:2,accepted:1,rejected:1});
  assert.equal(reviewStatus({...layer.objects[2],review_actor:'automated_qa'}),'Accepted · automated QA');
  layer.reviewPendingOnly=true;
  assert.deepEqual(layer.objects.filter(o=>layer.visible(o)).map(o=>o.id),['under','top']);
  assert.deepEqual(layer.objects,before);
});

test('generated SAM filter follows origin across review and refinement, but keeps new prompt masks', async()=>{
  const layer=scene(), geometry=layer.objects[0].geometry;
  layer.objects=[
    {id:'auto',status:'proposal',class_id:null,geometry,source:{kind:'sam2',operation:'automatic_mask_generation'}},
    {id:'reviewed',status:'accepted',class_id:'fruit',geometry,layer_id:'generation-1'},
    {id:'refined',status:'draft',class_id:'fruit',geometry,layer_id:'generation-1',source:{kind:'sam3'}},
    {id:'prompt',status:'draft',class_id:null,geometry,source:{kind:'sam2'}},
    {id:'manual',status:'draft',class_id:'fruit',geometry,source:{kind:'manual'}},
    {id:'yolo',status:'proposal',class_id:null,geometry,source:{kind:'yolo'}},
  ];
  const before=structuredClone(layer.objects);
  layer.select('auto');layer.hover='auto';layer.overlaps=['auto','prompt'];
  layer.hideGeneratedSAM=true;layer.pruneHiddenSelection();
  assert.equal(layer.selected,null);assert.equal(layer.hover,null);
  assert.deepEqual(layer.overlaps,['prompt']);
  assert.deepEqual((await layer.hits([5,5])).map(o=>o.id),['yolo','manual','prompt']);
  layer.select('reviewed');assert.equal(layer.selected,null);
  layer.hideAssigned=true;
  assert.deepEqual(layer.objects.filter(o=>layer.visible(o)).map(o=>o.id),['prompt','yolo']);
  layer.hideAssigned=false;layer.hideGeneratedSAM=false;
  assert.equal(layer.objects.filter(o=>layer.visible(o)).length,6);
  assert.deepEqual(layer.objects,before);
});

test('disabling saved mask fill removes the tint even while hovered or selected, without removing selection cues',()=>{
  const layer=scene(),calls=[];
  layer.ctx={save(){},restore(){},setLineDash(){},fill(){calls.push('fill');},strokeRect(){calls.push('bounds');},
    beginPath(){},moveTo(){},lineTo(){},closePath(){},stroke(){calls.push('outline');}};
  layer.transform={scale:1};layer.opacity=.36;layer.classes=[];
  const mask={id:'mask',class_id:null,status:'proposal',geometry:{type:'mask',rle:{size:[2,2],counts:[0,4]}}};
  layer.maskCache.get=()=>({path:{},bbox:[0,0,2,2]});
  layer.showMaskFill=true;layer.drawGeometry(mask);assert.deepEqual(calls,['fill']);
  layer.showMaskFill=false;calls.length=0;
  layer.drawGeometry(mask);assert.deepEqual(calls,[]);
  layer.hover='mask';layer.drawGeometry(mask);assert.deepEqual(calls,['bounds']);
  calls.length=0;layer.hover=null;layer.selected='mask';layer.drawGeometry(mask);
  assert.deepEqual(calls,['bounds']);
  calls.length=0;layer.drawGeometry(layer.objects[0]);assert.deepEqual(calls,['outline']);
  // A new model preview is deliberately kept readable before it is saved.
  calls.length=0;layer.selected=null;layer.drawGeometry(mask,true);assert.deepEqual(calls,['fill']);
  calls.length=0;layer.showMaskFill=true;layer.drawGeometry(mask);assert.deepEqual(calls,['fill']);
});

test('fill visibility does not change hit testing or annotation data',async()=>{
  const layer=scene(),before=structuredClone(layer.objects);
  layer.showMaskFill=false;
  assert.deepEqual((await layer.hits([5,5])).map(o=>o.id),['top','under']);
  layer.select('top');assert.equal(layer.selected,'top');
  assert.deepEqual(layer.objects,before);
});
