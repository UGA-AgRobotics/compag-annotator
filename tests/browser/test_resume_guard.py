"""Exercise installed Resume controls with explicit in-browser API test doubles.

No server mutation or provider execution. Real partial-job resume is covered by
test_addendum.py against the isolated synthetic provider server.
"""
from playwright.sync_api import expect


def test_restored_and_foreign_generation_resume_guards(page):
    expect(page.get_by_role('heading', name='Projects', exact=True)).to_be_visible()
    results = page.evaluate("""async () => {
        const {layerControls} = await import('/assets/generation.js');
        const valid = {
            id:'own-job', kind:'generate', status:'failed',
            payload:{project_id:'own-project', generation_id:'own-generation',
                layers:[{id:'own-layer', image_id:'own-image'}]}
        };
        const foreign = structuredClone(valid);
        foreign.id = 'foreign-job'; foreign.payload.project_id = 'foreign-project';
        const cases = [
            {name:'restored', layer:{resume_available:false,
                resume_unavailable_reason:'Restored job is provenance only.'}},
            {name:'foreign-project', detail:foreign},
            {name:'wrong-kind', detail:{...valid,kind:'train'}},
            {name:'wrong-layer', detail:{...valid,payload:{...valid.payload,
                layers:[{id:'other-layer',image_id:'own-image'}]}}},
            {name:'wrong-image', detail:{...valid,payload:{...valid.payload,
                layers:[{id:'own-layer',image_id:'other-image'}]}}},
            {name:'wrong-generation', detail:{...valid,payload:{...valid.payload,
                generation_id:'other-generation'}}},
            {name:'complete-job', detail:{...valid,status:'complete'}},
            {name:'valid'},
            {name:'fallback-project-filter', layer:{job_id:null}, jobs:[foreign,valid]},
            {name:'fallback-fresh-verification', layer:{job_id:null},
                jobs:[valid],detail:foreign},
            {name:'foreign-fallback-only', layer:{job_id:null}, jobs:[foreign]}
        ];
        const output = [];
        for (const test of cases) {
            const calls = [], shown = [];
            const editor = {
                project:{id:'own-project'}, image:{id:'own-image'},
                generationLayers:[{id:'own-layer',image_id:'own-image',status:'partial',
                    generation_id:'own-generation',job_id:'own-job',
                    completed_tiles:1,total_tiles:2,...test.layer}],
                layer:{hiddenLayers:new Set()},
                app:{api:{
                    get:async path => {calls.push(['GET',path]);
                        return path === '/api/jobs' ? test.jobs : test.detail || valid;},
                    post:async path => {calls.push(['POST',path]); return {id:'retried'};}
                },showJob:job => shown.push(job.id)}
            };
            document.body.replaceChildren();
            const notice = document.createElement('div'); notice.id = 'notice';
            document.body.append(notice,layerControls(editor));
            const button = [...document.querySelectorAll('button')]
                .find(b => b.textContent === 'Resume unfinished tiles');
            const disabled = button.disabled;
            const reason = document.body.textContent;
            button.click();
            if (!disabled) {
                for (let attempt=0;button.disabled && attempt<100;attempt++)
                    await new Promise(resolve => setTimeout(resolve,5));
                if (button.disabled) throw Error('Resume test did not settle');
            }
            output.push({name:test.name,disabled,reason,calls,shown,error:notice.textContent});
        }
        return output;
    }""")
    by_name = {item['name']: item for item in results}
    restored = by_name['restored']
    assert restored['disabled'] and not restored['calls']
    assert 'Restored job is provenance only.' in restored['reason']
    for name in ('foreign-project', 'wrong-kind', 'wrong-layer', 'wrong-image',
                 'wrong-generation', 'complete-job', 'fallback-fresh-verification',
                 'foreign-fallback-only'):
        result = by_name[name]
        assert not any(method == 'POST' for method, _ in result['calls']), result
        assert result['error'] and not result['shown'], result
    assert by_name['valid']['calls'] == [
        ['GET', '/api/jobs/own-job'], ['POST', '/api/jobs/own-job/retry']]
    assert by_name['fallback-project-filter']['calls'] == [
        ['GET', '/api/jobs'], ['GET', '/api/jobs/own-job'],
        ['POST', '/api/jobs/own-job/retry']]
    assert by_name['valid']['shown'] == ['retried']
    assert by_name['fallback-project-filter']['shown'] == ['retried']
