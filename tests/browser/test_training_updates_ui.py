"""Render actual Jobs components with explicit result fixtures, without model execution."""
from playwright.sync_api import expect


def test_weight_update_receipt_is_visible_without_claiming_accuracy_or_upgrading_old_jobs(page, tmp_path):
    expect(page.get_by_role('heading', name='Projects', exact=True)).to_be_visible()
    page.evaluate("""async () => {
        const {jobCard} = await import('/assets/jobs.js');
        const app = {navigate: async () => {}};
        const common = {kind:'train',status:'complete',created_at:'synthetic test',progress:{},log:[]};
        const old = {...common,id:'historical-job',result:{metrics:{'metrics/mAP50(M)':0}}};
        const verified = {...common,id:'measured-job',result:{metrics:{'metrics/mAP50(M)':0},
            optimization:{passed:true,positive_lr_steps:9,learnable_weights_changed:true}}};
        document.querySelector('#main').replaceChildren(jobCard(app,old),jobCard(app,verified));
    }""")
    expect(page.get_by_test_id('training-update-check')).to_have_count(1)
    expect(page.locator('[data-job-id="historical-job"]').get_by_test_id('training-update-check')).to_have_count(0)
    message = page.locator('[data-job-id="measured-job"]').get_by_test_id('training-update-check')
    expect(message).to_contain_text('9 optimizer steps with a nonzero learning rate')
    expect(message).to_contain_text('Learnable weights changed')
    expect(message).to_contain_text('Check validation metrics to assess prediction quality')
    page.screenshot(path=str(tmp_path / 'verified-updates-and-legacy-job.png'), full_page=True)
