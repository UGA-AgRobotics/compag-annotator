import test from 'node:test';
import assert from 'node:assert/strict';
import { incompatibleTrainingObjects } from '../../src/compag_annotator/web/assets/training-exclusions.js';
import { reviewStatus } from '../../src/compag_annotator/web/assets/review-status.js';

test('only incompatible geometry with object IDs is selectable, once per object', () => {
  const result = incompatibleTrainingObjects({errors: [
    {annotation_id:'a', code:'invalid_annotation_geometry'},
    {annotation_id:'a', code:'invalid_annotation_geometry'},
    {annotation_id:'b', code:'box_only_annotation'},
    {annotation_id:'c', code:'unreviewed_annotation'},
    {annotation_id:'d', code:'invalid_class'},
    {code:'missing_validation_images'},
  ]});
  assert.deepEqual(result.map(r => r.annotation_id), ['a','b']);
  assert.deepEqual(incompatibleTrainingObjects({}), []);
});

test('training exclusions are visible without changing the review decision', () => {
  assert.equal(reviewStatus({status:'accepted', training_excluded:true}), 'Accepted · Excluded from training');
  assert.equal(reviewStatus({status:'accepted'}), 'Accepted');
  assert.equal(reviewStatus({status:'accepted', review_actor:'automated_qa', training_excluded:true}),
    'Accepted · automated QA · Excluded from training');
});
