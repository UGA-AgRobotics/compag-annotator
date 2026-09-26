import test from 'node:test';
import assert from 'node:assert/strict';
import { imageReviewPlan, imageTrainingSummary } from '../../src/compag_annotator/web/assets/image-review.js';

test('review distinguishes labeled drafts, accepted masks and unassigned/archived classes without geometry', () => {
  const p={classes:[{id:'one'},{id:'old',archived:true}]},im={role:'validation'};
  const rows=[
    {id:'1',status:'draft',class_id:'one'},
    {id:'2',status:'accepted',class_id:'one'},
    {id:'3',status:'proposal',class_id:null},
    {id:'4',status:'draft',class_id:'old'},
    {id:'5',status:'rejected',class_id:null},
    {id:'6',status:'superseded',class_id:null},
  ];
  const before=structuredClone(rows);
  rows.forEach(r=>Object.defineProperty(r,'geometry',{get(){throw Error('Do not load geometry');}}));
  const result=imageReviewPlan(p,im,rows);
  assert.equal(result.pending.length,3);assert.equal(result.accepted,1);assert.equal(result.unlabeled,2);
  assert.equal(result.negative,false);assert.equal(result.eligibleRole,true);
  assert.deepEqual(rows,before);
  assert.equal(imageReviewPlan(p,{role:'excluded'},[]).eligibleRole,false);
  assert.equal(imageReviewPlan(p,im,rows.slice(4)).negative,true);
});
test('training summary counts reviewed images separately from roles and excludes test/removed images', () => {
  const result=imageTrainingSummary({images:[
    {role:'train',complete:true},{role:'pool',complete:false},{role:'validation',complete:false},
    {role:'test',complete:true},{role:'train',complete:true,removed:true},
  ]});
  assert.deepEqual(result,{training:2,validation:1,reviewedTraining:1,reviewedValidation:0});
});
