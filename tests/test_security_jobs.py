import io,json,stat,zipfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from compag_annotator.app import create_app
from compag_annotator.storage.files import safe_extract

def test_zip_symlink_rejected(tmp_path):
    path=tmp_path/'malicious.zip';entry=zipfile.ZipInfo('link');entry.create_system=3;entry.external_attr=(stat.S_IFLNK|0o777)<<16
    with zipfile.ZipFile(path,'w') as z:z.writestr(entry,'/etc/passwd')
    with pytest.raises(ValueError):safe_extract(path,tmp_path/'out')

def test_zip_limits_before_extraction(tmp_path):
    path=tmp_path/'huge.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('a.txt',b'x'*2000)
    with pytest.raises(ValueError):safe_extract(path,tmp_path/'out',limit=1000)
    assert not (tmp_path/'out').exists()

def test_post_token_content_length_and_cross_site(tmp_path):
    app=create_app(tmp_path/'app',token='private-test-token',qa_mode=True)
    with TestClient(app) as client:
        for headers in [{'X-Compag-Token':'wrong'},{'Sec-Fetch-Site':'cross-site','X-Compag-Token':'private-test-token'}]:
            assert client.post('/api/projects',json={'name':'x'},headers=headers).status_code==403
        assert client.post('/api/projects',content=b'{}',headers={'X-Compag-Token':'private-test-token','Content-Length':str(300*1024**2)}).status_code==413
        assert client.get('/api/jobs/../../etc/passwd/artifact').status_code in (400,404)
        assert client.get('/api/session').headers['Content-Security-Policy'].find("frame-ancestors 'none'")>=0

def test_sam3_checkpoint_network_action_forbidden_before_provider(tmp_path):
    app=create_app(tmp_path/'app',token='q',qa_mode=True)
    with TestClient(app) as client:
        result=client.post('/api/models/install',json={'provider':'sam3','architecture':'sam3-image','consent':True,'download_weights':True},headers={'X-Compag-Token':'q'})
        assert result.status_code==400 and 'disabled' in result.json()['detail']
        assert app.state.service.jobs.list()==[]
