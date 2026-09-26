"""Bounded geometry scheduling using explicit small, synthetic unit fixtures."""

from playwright.sync_api import expect


def test_exact_hit_geometry_bypasses_background_mask_queue(page):
    expect(page.get_by_role('heading', name='Projects', exact=True)).to_be_visible()
    result = page.evaluate("""async () => {
        const {MaskStore} = await import('/assets/masks.js');
        const {AnnotationCanvas} = await import('/assets/canvas.js');
        const started = [], release = new Map();
        const geometry = {type:'mask',rle:{size:[2,2],counts:[0,4]}};
        const store = new MaskStore(object => new Promise(resolve => {
            started.push(object.id); release.set(object.id,resolve);
        }));
        const objects = Array.from({length:1000},(_,i) => ({
            id:String(i),revision:1,geometry:{type:'mask',ref:String(i)},bbox:[0,0,2,2]
        }));
        objects.forEach(object => store.ensure(object));
        const target = objects.at(-1);
        const canvas = {objects:[target],maskCache:store,
            visible:()=>true,near:()=>true,hit:(o,p)=>store.hit(o,p)};
        // Call the production hit path, which must promote an already queued
        // mask without duplicating its request or increasing concurrency.
        const hit = AnnotationCanvas.prototype.hits.call(canvas,[1,1]);
        const initial = [...started];
        release.get('0')(geometry);
        for(let i=0;i<10 && !release.has('999');i++) await Promise.resolve();
        if (!release.has('999')) throw Error('Interactive mask remains behind background geometry');
        release.get('999')(geometry);
        const selected = await hit;
        await store.ensure(target,true);
        return {initial,started,selected:selected.map(o=>o.id),active:store.active,
            queued:store.queue.length,loadsOfTarget:started.filter(id=>id==='999').length};
    }""")
    assert result['initial'] == ['0', '1', '2', '3']
    assert result['started'][4] == '999'
    assert result['selected'] == ['999']
    assert result['loadsOfTarget'] == 1
    assert result['active'] <= 4 and result['queued'] > 990
