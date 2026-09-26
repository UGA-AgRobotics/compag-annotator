"""User-approved project locations and portable copy-based backups."""
import copy,json,shutil,sqlite3,threading,zipfile
from pathlib import Path
from .projects import Project
from compag_annotator.storage.files import atomic,uid,now,safe_extract,safe_child,digest

class Catalog:
    def __init__(self,data_dir):
        self.root=Path(data_dir);self.file=self.root/'projects.json';self.lock=threading.RLock()
        self.root.mkdir(parents=True,exist_ok=True)
        if not self.file.exists():atomic(self.file,[])
    def records(self):return json.loads(self.file.read_text())
    def register(self,project):
        state=project.state();row={'id':state['id'],'name':state['name'],'path':str(project.path),'opened_at':now()}
        with self.lock:atomic(self.file,[r for r in self.records() if r['id']!=row['id']]+[row])
        return project
    def get(self,pid):
        row=next((r for r in self.records() if r['id']==pid),None)
        if not row:raise ValueError('Project is not registered; open its folder first')
        return Project(row['path'])
    def create(self,name,classes,path=None):
        chosen=Path(path).expanduser() if path else self.root/'projects'/uid()
        return self.register(Project.create(chosen,name,classes))
    def open(self,path):return self.register(Project(path))
    def backup(self,project,destination,include_images=True):
        destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
        stage=destination.parent/('backup-'+uid());stage.mkdir()
        try:
            with project.lock,project.connection() as db:
                out=sqlite3.connect(stage/'project.sqlite3');db.backup(out)
                state=json.loads(out.execute('SELECT body FROM state WHERE id=1').fetchone()[0])
                for image in state['images']:
                    if image.get('removed'):continue
                    if image['storage']=='reference' and include_images:
                        source=Path(image['original'])
                        if not source.is_file() or digest(source)!=image['sha256']:raise ValueError('Linked original missing/changed; relink before including it')
                        relative=f"media/{image['id']}/original{source.suffix.lower()}";target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
                        image['original']=relative;image['storage']='copy'
                for model in state.get('models',[]):
                    model.pop('path',None);model['weights_included']=False
                state['active_model_available_after_restore']=False
                out.execute('UPDATE state SET body=? WHERE id=1',(json.dumps(state),));out.commit();out.close()
                for folder in ['masks']+(['media'] if include_images else []):
                    origin=project.path/folder
                    if origin.exists():shutil.copytree(origin,stage/folder,dirs_exist_ok=True)
                shutil.copy2(project.path/'.compag-annotator.json',stage/'.compag-annotator.json')
            atomic(stage/'native_manifest.json',{'schema':'compag-native/v1','original_project_id':state['id'],'includes_images':include_images,'model_weights_included':False,'created_at':now(),'files':{str(p.relative_to(stage)):digest(p) for p in stage.rglob('*') if p.is_file()}})
            temp=destination.with_suffix('.partial')
            with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
                for p in sorted(stage.rglob('*')):
                    if p.is_file():z.write(p,p.relative_to(stage))
            temp.replace(destination)
            return {'artifact':str(destination),'sha256':digest(destination),'format':'native','images_included':include_images,'weights_included':False}
        finally:shutil.rmtree(stage)
    def restore(self,archive):
        target=self.root/'projects'/uid();target.parent.mkdir(parents=True,exist_ok=True);safe_extract(archive,target)
        try:
            manifest=json.loads((target/'native_manifest.json').read_text())
            if manifest.get('schema')!='compag-native/v1':raise ValueError('Unsupported native backup')
            for name,h in manifest['files'].items():
                if digest(safe_child(target,name))!=h:raise ValueError('Backup checksum mismatch')
            project=Project(target)
            # Clone gets a new workspace identity; annotation/image/class/history IDs remain.
            with project.edit(None,'backup_restore','import') as (db,state):
                state['restored_from_project_id']=state['id'];state['id']=uid();state['active_model_id']=None
                for layer in state.get('generation_layers',[]):
                    layer['resume_available']=False
                    layer['resume_unavailable_reason']='This restored project preserves proposals and history, but generation job artifacts belong to the source project. Start a new generation here.'
                for image in state['images']:
                    if not (target/image['path']).exists():image['media_unavailable']=True
            atomic(target/'.compag-annotator.json',{'schema_version':1,'id':project.state()['id']})
            return self.register(project)
        except BaseException:shutil.rmtree(target);raise
