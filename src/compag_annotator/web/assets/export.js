import { el, button, select, field, check, card, row, heading, details, projectPath } from './ui.js';
import { importFormFor } from './imports.js';

export const datasetPresets = [
  {id:'coco',format:'coco',title:'COCO — instance segmentation',description:'Recommended for exact SAM masks, including holes and separate parts. JSON with classes, instances and RLE masks.',files:'annotations.json · manifest.json'},
  {id:'yolo_seg',format:'yolo_seg',title:'YOLO — instance segmentation',description:'One polygon label per object, with dataset.yaml and existing train/validation/test splits. Complex masks are checked during conversion.',files:'labels/*.txt · dataset.yaml · class_map.json · train.txt / val.txt / test.txt'},
  {id:'yolo_box',format:'yolo_box',title:'YOLO — object detection',description:'Bounding boxes for detector training. This format does not retain mask outlines.',files:'labels/*.txt · dataset.yaml · class_map.json · train.txt / val.txt / test.txt'},
  {id:'png_semantic',format:'png',png_variant:'semantic',title:'PNG — semantic segmentation',description:'One class-ID mask per image for pixel classification. Background is 0; class values start at 1. Individual objects of the same class become one class region.',files:'masks/*.png · mapping.json'},
  {id:'png_instances',format:'png',png_variant:'per_instance',title:'PNG — separate instance masks',description:'One binary PNG per object. Retains exact mask pixels and overlapping objects. Useful for custom segmentation data loaders.',files:'masks/<image>/<object>.png · mapping.json'},
  {id:'png_ids',format:'png',png_variant:'instance_id',title:'PNG — instance-ID masks',description:'One 16-bit mask per image, with an ID for each object. A single pixel can belong to only one instance.',files:'masks/*.png · mapping.json'},
  {id:'voc',format:'voc',title:'Pascal VOC — object detection',description:'One XML file per image containing object classes and bounding boxes. Mask outlines are omitted.',files:'annotations/*.xml'},
  {id:'labelme',format:'labelme',title:'LabelMe — polygons and rectangles',description:'Editable annotation JSON for LabelMe-compatible workflows. Mask holes may need an explicitly allowed approximation.',files:'images/*.json'},
];
const importFormats = [['coco','COCO instances'],['yolo_seg','YOLO segmentation'],['yolo_box','YOLO detection'],['voc','Pascal VOC'],['labelme','LabelMe'],['png','PNG masks + metadata']];
const count = (n,word) => `${n} ${word}${n===1?'':'s'}`;
const bytes = value => value < 1048576 ? `${(value/1024).toFixed(1)} KiB` : `${(value/1048576).toFixed(1)} MiB`;

export async function exportPage(app) {
  const project=app.project, base=projectPath(project.id), selected=[...app.selectedImages];
  const abort=new AbortController();let alive=true,sequence=0,preview=null,previewKey=null;
  app.cleanup=()=>{alive=false;sequence++;abort.abort();};
  const preset=select('training-format',datasetPresets.map(p=>[p.id,p.title]),'coco');
  const annotationScope=select('annotation-scope',[
    ['reviewed','Reviewed dataset — accepted objects on reviewed images'],
    ['labeled','Labeled objects — include drafts, omit unassigned masks'],
  ],'reviewed');
  const images=check('dataset-images','Include image copies (larger, portable dataset)',false);
  const scope=select('dataset-scope',[
    ['all','All project images'],['selected',`Selected images (${selected.length})`],
    ['round','One review round'],['cumulative','Reviewed cumulative training images'],
  ],selected.length?'selected':'all');
  const roundId=select('dataset-round',[['','Choose a round'],...project.rounds.map(r=>[r.id,`Round ${r.number}`])]);
  const roundField=field('Review round',roundId);
  const allow=check('dataset-conversion','Allow approximate polygons and record geometry changes',false);
  const allowLabel=allow.querySelector('span');
  const overlap=select('dataset-overlap',[['error','Keep overlaps unresolved — block this export'],['first','Earlier object keeps overlapping pixels'],['last','Later object keeps overlapping pixels']],'error');
  const overlapField=field('Overlapping objects in one PNG',overlap);
  const conversion=el('div',{},allow,overlapField);
  const description=el('p'),fileList=el('small');
  const summary=el('section',{'aria-label':'Export preview','aria-live':'polite'});
  const output=el('section',{'aria-label':'Export result','aria-live':'polite'});
  const chosen=()=>datasetPresets.find(p=>p.id===preset.value);
  const request=()=>({format:chosen().format,png_variant:chosen().png_variant||'per_instance',
    annotation_scope:annotationScope.value,scope:scope.value,image_ids:selected,
    round_id:scope.value==='round'?roundId.value:undefined,
    include_images:images.querySelector('input').checked,
    allow_lossy:!allow.hidden&&allow.querySelector('input').checked,
    conflict_policy:!overlapField.hidden?overlap.value:'error'});
  const key=()=>JSON.stringify(request());
  const create=button('Create dataset ZIP',async()=>{
    if(!preview?.ready||previewKey!==key())throw Error('Check the export with the current settings first.');
    controls.disabled=true;checkButton.disabled=true;
    const submitted={...request(),expected_revision:preview.project_revision};
    output.replaceChildren(el('p',{},'Preparing your dataset…'));
    try {
      const job=await app.api.post(base+'/exports',submitted);
      const complete=await app.api.waitJob(job,{signal:abort.signal,onProgress:j=>{
        if(!alive)return;
        const p=j.progress||{};
        output.replaceChildren(el('p',{},p.stage||'Preparing dataset…'),el('progress',{'aria-label':'Export progress',...(p.total?{value:p.completed,max:p.total}:{})}),
          el('small',{},'You can leave this page; the export remains available in Jobs.'));
      }});
      if(!alive)return;
      const r=complete.result;
      output.replaceChildren(el('h2',{},'Your dataset is ready'),
        el('p',{},`${count(r.summary.image_count,"image")} · ${count(r.summary.annotation_count,"annotation")} · ZIP ${bytes(r.archive_bytes)}`),
        el('p',{},`${r.images_included?'Image copies included':'Annotations only — image copies excluded'} · No project history or model weights.`),
        el('a',{class:'button primary',href:`/api/jobs/${complete.id}/artifact`,download:true},'Download dataset ZIP'),
        details('Export notes',r.report));
    } catch(error) {
      if(error.name==='AbortError')return;
      if(alive)output.replaceChildren(el('p',{class:'inline-warning',role:'alert'},error.message),
        el('p',{},'No annotations were changed. For complex mask shapes, try COCO or separate instance PNG masks.'),
        button('Open Jobs',()=>app.navigate('jobs')));
    } finally {
      controls.disabled=false;checkButton.disabled=false;
    }
  },'primary');
  create.hidden=true;
  const invalidate=()=>{
    sequence++;preview=null;previewKey=null;create.hidden=true;output.replaceChildren();
    summary.replaceChildren(el('p',{},'Check the export to see exactly which images and annotations will be included.'));
  };
  function configure() {
    const p=chosen();description.textContent=p.description;
    fileList.textContent=`Inside the ZIP: ${p.files} · README.md · image_index.json`;
    allow.querySelector('input').checked=false;overlap.value='error';
    allow.hidden=!['yolo_seg','yolo_box','voc','labelme'].includes(p.format)&&p.png_variant!=='semantic';
    const label=['yolo_box','voc'].includes(p.format)?'Export bounding boxes only; omit mask outlines and allow box-coordinate rounding'
      :p.png_variant==='semantic'?'Combine same-class objects into semantic class regions (instance identities are omitted)'
      :'Allow approximate polygons and record geometry changes';
    if(allowLabel)allowLabel.textContent=label;
    overlapField.hidden=!(p.format==='png'&&p.png_variant!=='per_instance');
    conversion.hidden=allow.hidden&&overlapField.hidden;
    invalidate();
  }
  const checkButton=button('Check export',async()=>{
    const current=++sequence,payload=request(),signature=key();
    summary.replaceChildren(el('p',{},'Checking images and class assignments…'));create.hidden=true;
    try {
      const result=await app.api.post(base+'/exports/preview',payload,abort.signal);
      if(!alive||current!==sequence||signature!==key())return;
      preview=result;previewKey=signature;
      const excludedLabels={excluded_role_images:'Images marked excluded',rejected_or_superseded_objects:'Rejected or merged-away objects',unassigned_or_archived_class_objects:'Objects without an active class',not_reviewed_objects:'Objects not eligible for reviewed export',incomplete_images:'Images not marked reviewed',unlabeled_incomplete_images:'Unfinished images without labeled objects'};
      summary.replaceChildren(el('h2',{},result.ready?'Ready to export':'Export needs attention'),
        row(el('strong',{},count(result.image_count,"image")),el('strong',{},count(result.annotation_count,"annotation")),el('span',{},`${result.class_counts.length} ${result.class_counts.length===1?"class":"classes"}`)),
        el('p',{},`${result.images_included?'Annotations and matching image copies':'Annotations only'} · No project database, Undo/Redo history or model weights.`),
        el('p',{},`Train: ${result.split_counts.train} · Validation: ${result.split_counts.val} · Test: ${result.split_counts.test}`),
        el('p',{},result.annotation_scope==='labeled'?`Draft export · ${count(result.draft_count,"unreviewed object")} · ${count(result.incomplete_image_count,"incomplete image")}`:'Accepted objects on images marked reviewed'),
        ...result.errors.map(error=>el('p',{class:'inline-warning'},error)),
        ...result.warnings.map(warning=>el('p',{},warning)),
        el('details',{},el('summary',{},'Class counts and skipped items'),
          el('ul',{},result.class_counts.map(c=>el('li',{},`${c.name}: ${c.count}`))),
          el('ul',{},Object.entries(result.excluded).filter(([,n])=>n).map(([reason,n])=>el('li',{},`${excludedLabels[reason]||reason}: ${n}`)))),
        ...(result.image_count===0&&annotationScope.value==='reviewed'?[button('Use current labeled objects instead',()=>{
          annotationScope.value='labeled';invalidate();
        })]:[]),
        ...(!result.ready?[button('Open Images to review',()=>app.navigate('images'))]:[]));
      create.hidden=!result.ready;
    } catch(error) {
      if(error.name!=='AbortError'&&alive&&current===sequence)
        summary.replaceChildren(el('p',{class:'inline-warning',role:'alert'},error.message));
    }
  });
  const controls=el('fieldset',{class:'export-controls'},
    field('1. Training format',preset),description,fileList,
    field('2. Which annotations?',annotationScope),
    el('small',{},'Class assignment is different from acceptance. Draft exports never mark objects reviewed. Unassigned SAM proposals are not training labels.'),
    el('h2',{},'3. Include images?'),images,
    el('small',{},'Leave unchecked for a small annotations-only ZIP. Check to package matching images for another computer.'),
    el('details',{},el('summary',{},'Image selection'),field('Image scope',scope),roundField),
    conversion);
  preset.addEventListener('change',configure);
  for(const control of [annotationScope,images,allow,overlap,roundId])control.addEventListener('change',invalidate);
  scope.addEventListener('change',()=>{roundField.hidden=scope.value!=='round';invalidate();});
  roundField.hidden=true;configure();
  const backupImages=check('backup-images','Include original image files in backup',false);
  const backup=el('details',{},el('summary',{},'Project backup — continue editing later'),
    el('p',{},'Native backup saves the whole project, every proposal and Undo/Redo history. It can be large. Training-format and annotation-selection settings above do not apply. Model weights are excluded.'),backupImages,
    button('Create project backup',async()=>app.showJob(await app.api.post(base+'/exports',{format:'native',include_images:backupImages.querySelector('input').checked}))));
  return el('div',{class:'dataset-export-page'},heading('Export annotations','Choose a training format, check the contents, then download your ZIP.'),
    card('Training dataset',controls,row(checkButton,create),summary,output),
    card('Backup and import',backup,el('details',{},el('summary',{},'Import existing annotations'),importFormFor(app,importFormats))));
}
