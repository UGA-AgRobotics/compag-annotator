"""Durable serial job queue; provider processes are cancellable by callbacks."""
import concurrent.futures,copy,json,threading,traceback
from pathlib import Path
from compag_annotator.storage.files import atomic,uid,now

class AwaitingReview(ValueError):pass

class JobRunner:
    def __init__(self,data_dir):
        self.root=Path(data_dir)/'jobs';self.root.mkdir(parents=True,exist_ok=True)
        self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='compag-job')
        self.lock=threading.RLock();self.cancels={};self.functions={}
        for p in self.root.glob('*/job.json'):
            job=json.loads(p.read_text())
            if job['status'] in ('running','queued'):
                job.update(status='interrupted',error='Application stopped before this job completed; explicit retry or checkpoint resume is available',finished_at=now());atomic(p,job)
    def get(self,jid):
        if not isinstance(jid,str) or len(jid)!=32 or not all(c in '0123456789abcdef' for c in jid):raise ValueError('Invalid job identifier')
        path=self.root/jid/'job.json'
        if not path.exists():raise ValueError('Job not found')
        return json.loads(path.read_text())
    def list(self):return sorted((json.loads(p.read_text()) for p in self.root.glob('*/job.json')),key=lambda j:j['created_at'],reverse=True)
    def update(self,jid,**values):
        with self.lock:
            job=self.get(jid);job.update(values);job['updated_at']=now();atomic(self.root/jid/'job.json',job)
        return job
    def submit(self,kind,payload,function,key=None):
        with self.lock:
            if key:
                old=next((j for j in self.list() if j.get('idempotency_key')==key and j['status'] in ('queued','running','complete')),None)
                if old:return old
            jid=uid();directory=self.root/jid;directory.mkdir();cancel=threading.Event();self.cancels[jid]=cancel;self.functions[jid]=function
            job={'id':jid,'kind':kind,'status':'queued','payload':payload,'idempotency_key':key,'created_at':now(),'progress':{'stage':'queued'},'result':None,'error':None,'log':[]}
            atomic(directory/'job.json',job)
            self.executor.submit(self._run,jid,function,cancel)
            return job
    def _run(self,jid,function,cancel):
        def progress(value):
            if cancel.is_set():raise InterruptedError('Job cancelled')
            self.update(jid,progress=value)
        try:
            if cancel.is_set():raise InterruptedError('Job cancelled before starting')
            self.update(jid,status='running',started_at=now(),progress={'stage':'starting'})
            result=function(self.root/jid,progress,cancel.is_set)
            if cancel.is_set():raise InterruptedError('Job cancelled')
            self.update(jid,status='complete',result=result,finished_at=now(),progress={'stage':'complete'})
        except AwaitingReview as error:self.update(jid,status='awaiting_review',error=str(error),result=getattr(error,'result',None),finished_at=now())
        except InterruptedError as error:self.update(jid,status='cancelled',error=str(error),result=getattr(error,'result',None),finished_at=now())
        except BaseException as error:
            atomic(self.root/jid/'failure.log',traceback.format_exc().encode())
            self.update(jid,status='failed',error=str(error),finished_at=now(),log=[type(error).__name__+': '+str(error)],result={'loss_report':getattr(error,'loss_report',None)} if hasattr(error,'loss_report') else getattr(error,'result',None))
    def cancel(self,jid):
        job=self.get(jid)
        if job['status'] not in ('queued','running'):return job
        if jid in self.cancels:self.cancels[jid].set()
        return self.update(jid,progress={'stage':'cancelling'})
    def shutdown(self):
        for event in self.cancels.values():event.set()
        self.executor.shutdown(wait=False,cancel_futures=True)
