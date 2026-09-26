"""Atomic local files and bounded archive handling; never follow archive links."""
from __future__ import annotations
import hashlib,json,os,stat,tempfile,zipfile
from pathlib import Path

def uid():
    import uuid
    return uuid.uuid4().hex

def now():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def json_bytes(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not isinstance(data,bytes):data=json_bytes(data)
    fd,name=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
        dfd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(dfd)
        finally:os.close(dfd)
    finally:
        if os.path.exists(name):os.unlink(name)

def safe_child(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('Path escapes the selected project')
    return path

def safe_extract(archive,destination,limit=4*1024**3,max_files=50000):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=False)
    try:
        with zipfile.ZipFile(archive) as z:
            infos=z.infolist();total=sum(i.file_size for i in infos)
            if len(infos)>max_files or total>limit:raise ValueError('Archive exceeds configured extraction limit')
            seen=set()
            for i in infos:
                p=Path(i.filename)
                if p.is_absolute() or '..' in p.parts or '\\' in i.filename or stat.S_ISLNK(i.external_attr>>16):raise ValueError('Unsafe archive entry')
                target=safe_child(dest,p)
                if str(target) in seen:raise ValueError('Duplicate archive entry')
                seen.add(str(target))
                if i.is_dir():target.mkdir(parents=True,exist_ok=True);continue
                if i.file_size>128*1024**2 and i.compress_size and i.file_size/i.compress_size>1000:raise ValueError('Suspicious archive expansion')
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(i) as source,target.open('xb') as out:
                    count=0
                    for block in iter(lambda:source.read(1024**2),b''):
                        count+=len(block)
                        if count>i.file_size:raise ValueError('Archive size mismatch')
                        out.write(block)
        return dest
    except BaseException:
        import shutil
        shutil.rmtree(dest)
        raise
