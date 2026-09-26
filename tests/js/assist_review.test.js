import test from 'node:test';
import assert from 'node:assert/strict';
import { isSAMAssist, addAssistPoint, assistDrafts } from '../../src/compag_annotator/web/assets/assist-review.js';
import { AnnotationCanvas } from '../../src/compag_annotator/web/assets/canvas.js';

test('SAM Assist provenance excludes automatic, manual, YOLO and vector shapes', () => {
  const legacy={id:'one',geometry:{type:'mask'},source:{kind:'sam2'},status:'draft'};
  assert.ok(isSAMAssist(legacy));
  assert.ok(isSAMAssist({...legacy,source:{kind:'sam3',operation:'independent_point_assist'}}));
  assert.ok(isSAMAssist({...legacy,layer_id:'auto',source:{kind:'sam2',operation:'sam_assist'}}));
  assert.equal(isSAMAssist({...legacy,layer_id:'auto'}),false);
  assert.equal(isSAMAssist({...legacy,source:{kind:'sam2',operation:'automatic_mask_generation'}}),false);
  for(const kind of ['manual','yolo','import'])assert.equal(isSAMAssist({...legacy,source:{kind}}),false);
  assert.equal(isSAMAssist({...legacy,geometry:{type:'polygon'}}),false);
  assert.equal(assistDrafts({objects:[legacy,{...legacy,status:'accepted'},{...legacy,status:'proposal'}]}).length,1);
});

test('point placement is immediate and never reads saved mask geometry or visibility', () => {
  const e={alive:true,promptState:{mode:'independent',points:[],labels:[]},
    layer:{get objects(){throw Error('Saved mask scan is forbidden during point placement');},
      get maskCache(){throw Error('Geometry loading is forbidden during point placement');},requestRender(){}},
    invalidateAssist(redraw){assert.equal(redraw,false);},syncPrompts(redraw){assert.equal(redraw,false);},renderInspector(){}};
  const result=addAssistPoint(e,[10,10],1);
  assert.equal(result,undefined);
  assert.deepEqual(e.promptState.points,[[10,10]]);
});

test('more than 64 independent points remain in order without truncation', async () => {
  const e={alive:true,revision:2,promptState:{mode:'independent',points:[],labels:[]},
    layer:{objects:[],requestRender(){}},invalidateAssist(){},syncPrompts(){},renderInspector(){}};
  for(let i=0;i<130;i++)await addAssistPoint(e,[i,1],1);
  assert.equal(e.promptState.points.length,130);assert.deepEqual(e.promptState.points.at(-1),[129,1]);
  assert.equal(e.promptState.labels.length,130);
});

test('closed or saving editors do not receive new points', () => {
  for (const changes of [{alive:false},{busy:true},{failed:{error:'failed'}}]) {
    const e={alive:true,promptState:{mode:'independent',points:[],labels:[]},...changes};
    addAssistPoint(e,[1,1],1);assert.equal(e.promptState.points.length,0);
  }
});

test('display color overrides do not change the class or annotation', () => {
  const row={id:'one',class_id:'c'}, before=structuredClone(row);
  const layer={maskColors:new Map([['one','#112233']]),classes:[{id:'c',color:'#aabbcc'}]};
  assert.equal(AnnotationCanvas.prototype.color.call(layer,row),'#112233');
  layer.defaultMaskColor='#334455';assert.equal(AnnotationCanvas.prototype.color.call(layer,row),'#112233');
  layer.maskColors.clear();assert.equal(AnnotationCanvas.prototype.color.call(layer,row),'#334455');
  layer.defaultMaskColor=null;assert.equal(AnnotationCanvas.prototype.color.call(layer,row),'#aabbcc');
  assert.deepEqual(row,before);
});


test('rapid redraw requests paint once, direct redraw cancels pending frame, destroy cancels work', () => {
  const oldRAF=globalThis.requestAnimationFrame, oldCancel=globalThis.cancelAnimationFrame;
  const scheduled=new Map(); let id=0, paints=0;
  globalThis.requestAnimationFrame=fn=>{scheduled.set(++id,fn);return id;};
  globalThis.cancelAnimationFrame=key=>scheduled.delete(key);
  try {
    const layer=Object.assign(Object.create(AnnotationCanvas.prototype),{alive:true,render(){paints++;}});
    for(let i=0;i<100;i++)layer.requestRender();
    assert.equal(scheduled.size,1);const work=[...scheduled.values()][0];scheduled.clear();work();assert.equal(paints,1);
    layer.requestRender();AnnotationCanvas.prototype.render.call(layer);assert.equal(scheduled.size,0);
    layer.events={abort(){}};layer.observer={disconnect(){}};layer.maskCache={clear(){}};
    layer.requestRender();layer.destroy();assert.equal(scheduled.size,0);
  } finally {globalThis.requestAnimationFrame=oldRAF;globalThis.cancelAnimationFrame=oldCancel;}
});
