import test from 'node:test';
import assert from 'node:assert/strict';
import { nextReviewImage, openJobContext } from '../../src/compag_annotator/web/assets/workflow-guide.js';

test('next review wraps, skips reviewed/missing/excluded images and never mutates roles', () => {
  const p = {images: [{id:'first', complete:false, role:'validation'},
    {id:'done', complete:true}, {id:'current'}, {id:'missing', missing:true},
    {id:'excluded', role:'excluded'}]};
  const before = JSON.stringify(p);
  assert.equal(nextReviewImage(p, 'current').id, 'first');
  assert.equal(nextReviewImage(p).id, 'first');
  assert.equal(nextReviewImage({images:[{id:'only', complete:true}]}), undefined);
  assert.equal(nextReviewImage({images:[]}), undefined);
  assert.equal(JSON.stringify(p), before);
});

test('job navigation opens the owning project and exact image, with no write API', async () => {
  const seen = [];
  const app = {project:{id:'different'},
    async openProject(id) { seen.push(['project',id]); this.project={id,images:[{id:'target'}]}; },
    async refreshProject() {seen.push(['refresh']);},
    async annotate(id) {seen.push(['image',id]);},
    async navigate(route) {seen.push(['route',route]);},
  };
  await openJobContext(app, {payload:{project_id:'owner', layers:[{image_id:'target'}]}}, 'annotate');
  assert.deepEqual(seen, [['project','owner'],['refresh'],['image','target']]);
  seen.length = 0;
  await openJobContext(app, {payload:{project_id:'owner',image_id:'removed'}}, 'annotate');
  assert.deepEqual(seen, [['refresh'],['route','images']]);
});

test('a declined project switch or missing project binding cannot open the wrong image', async () => {
  const app = {project:{id:'other'}, async openProject(){},
    async refreshProject(){assert.fail('Unexpected refresh');},
    async annotate(){assert.fail('Wrong project image');},
    async navigate(){assert.fail('Unexpected navigation');},
  };
  await openJobContext(app, {payload:{project_id:'target',image_id:'image'}}, 'annotate');
  await assert.rejects(openJobContext(app, {payload:{}}, 'annotate'), /no project reference/);
});
