#!/usr/bin/env python3
"""Compare browser canvas pan work on an explicitly supplied, read-only mask scene.

Runs isolated QA servers. No inference, project changes or human-UAT claims.
The scene needs width, height, classes, and annotations with canonical geometry.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
from playwright.sync_api import sync_playwright

BENCHMARK = """async (scene) => {
  const {AnnotationCanvas} = await import('/assets/canvas.js');
  const canvas=document.createElement('canvas');
  canvas.style.cssText='position:fixed;top:0;left:0;width:1024px;height:768px;z-index:9999';
  document.body.append(canvas);
  const layer=new AnnotationCanvas(canvas,{});
  layer.record={width:scene.width,height:scene.height};
  const background=document.createElement('canvas');background.width=64;background.height=64;
  const bg=background.getContext('2d');bg.fillStyle='#eeeeee';bg.fillRect(0,0,64,64);
  layer.image=background;layer.objects=structuredClone(scene.annotations);layer.classes=scene.classes;
  layer.maskColors=new Map();layer.overlapColor='#ff40c8';layer.opacity=.36;
  const initial={x:10,y:10,scale:Math.min(1000/scene.width,740/scene.height)};
  layer.transform={...initial};
  layer.objects.forEach(o=>layer.maskCache.get(o));
  const frame=()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
  await frame();
  let calls=0, cpu=0;
  const render=layer.render.bind(layer);
  layer.render=()=>{const start=performance.now();render();cpu+=performance.now()-start;calls++;};
  const results=[];
  for(const overlap of [false,true]) {
    layer.showOverlap=overlap;
    for(let repeat=0;repeat<3;repeat++) {
      layer.transform={...initial};layer.gesture=null;layer.render();await frame();
      calls=0;cpu=0;
      const rect=canvas.getBoundingClientRect(),start=performance.now();
      layer.gesture={kind:'pan',start:[0,0],transform:{...initial}};
      for(let i=1;i<=24;i++)layer.move({clientX:rect.left+i,clientY:rect.top+i});
      // Real pan rendering in rc17 skips overlap during the gesture; its final
      // settled draw recomputes the overlap at the new transform.
      await frame();
      const burstCalls=calls,burstCpu=cpu;
      layer.gesture=null;layer.render();
      results.push({legacy_overlap_preference:overlap,repeat,pointer_events:24,
        burst_canvas_draws:burstCalls,burst_render_cpu_ms:burstCpu,
        final_draw_cpu_ms:cpu-burstCpu,total_render_cpu_ms:cpu,
        wall_ms:performance.now()-start,final_transform:{...layer.transform},
        overlap_scratch_canvases:layer.overlapCache?2:0});
    }
  }
  layer.destroy();canvas.remove();return results;
}"""


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    scene=json.loads(args.scene.read_text());results={}
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True)
        try:
            for name,source in [('baseline',args.baseline),('candidate',args.candidate)]:
                with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
                env={**os.environ,'PYTHONPATH':str(source.resolve()/'src')}
                with (args.output/(name+'.log')).open('w') as log:
                    process=subprocess.Popen([sys.executable,'-m','compag_annotator.cli','serve','--no-browser',
                        '--qa-mode','--port',str(port),'--data-dir',str((args.output/(name+'-app')).resolve())],
                        cwd=args.output,env=env,stdout=log,stderr=subprocess.STDOUT)
                try:
                    url=f'http://127.0.0.1:{port}'
                    for _ in range(100):
                        try:
                            with urllib.request.urlopen(url+'/api/session',timeout=.5):break
                        except OSError:time.sleep(.1)
                    else:raise RuntimeError('QA server did not start')
                    page=browser.new_page(viewport={'width':1280,'height':900})
                    page.goto(url);results[name]=page.evaluate(BENCHMARK,scene);page.close()
                finally:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
        finally:browser.close()
    assert all(r['burst_canvas_draws']==24 for r in results['baseline'])
    assert all(r['burst_canvas_draws']==1 and r['overlap_scratch_canvases']==0 for r in results['candidate'])
    assert [r['final_transform'] for r in results['baseline']]==[r['final_transform'] for r in results['candidate']]
    receipt={'status':'PASS','mask_count':len(scene['annotations']),'image_size':[scene['width'],scene['height']],
             'scope':'Controlled 24-event pan bursts on stored mask geometry and a plain background; not whole-app FPS or inference timing.',
             'live_projects_modified':False,'results':results}
    (args.output/'RESULTS.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
