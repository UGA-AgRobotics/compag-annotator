"""Real browser and isolated local API fixtures; all review actors are automated QA."""
from __future__ import annotations
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import pytest
from playwright.sync_api import sync_playwright


@pytest.fixture(scope="session")
def app_url(tmp_path_factory):
    root=Path(__file__).resolve().parents[2]
    data=tmp_path_factory.mktemp("browser-app-data")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
    env={**os.environ,"PYTHONPATH":str(root/"src")}
    executable=os.environ.get('COMPAG_TEST_INSTALLED_PYTHON',sys.executable)
    working_directory=root
    if os.environ.get('COMPAG_TEST_INSTALLED_PYTHON'):
        env.pop('PYTHONPATH',None)
        working_directory=data.parent
    process=subprocess.Popen([executable,"-m","compag_annotator.cli","serve","--no-browser","--qa-mode","--port",str(port),"--data-dir",str(data)],env=env,cwd=working_directory,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    url=f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(url+"/api/session",timeout=.5):break
            except OSError:
                if process.poll() is not None:raise RuntimeError(process.stdout.read().decode())
                time.sleep(.1)
        else:raise RuntimeError("Isolated browser app did not start")
        yield url
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(browser,app_url,tmp_path):
    context=browser.new_context(viewport={"width":1440,"height":1000})
    page=context.new_page();page.set_default_timeout(10000)
    errors=[];page.on("pageerror",lambda error:errors.append(str(error)))
    page.goto(app_url)
    yield page
    page.screenshot(path=str(tmp_path/"final-page.png"),full_page=True)
    context.close()
    assert not errors, f"Uncaught browser errors: {errors}"


@pytest.fixture(scope="session")
def mock_provider_app_url(tmp_path_factory):
    """Own synthetic data; production server with an explicitly mocked provider."""
    root=Path(__file__).resolve().parents[2]
    data=tmp_path_factory.mktemp("explicit-provider-mock-data")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
    executable=os.environ.get('COMPAG_TEST_INSTALLED_PYTHON',sys.executable)
    env={**os.environ,"PYTHONPATH":str(root/'src')}
    if os.environ.get('COMPAG_TEST_INSTALLED_PYTHON'):env.pop('PYTHONPATH',None)
    process=subprocess.Popen([executable,str(root/'tests/browser/provider_qa_server.py'),'--data-dir',str(data),'--port',str(port)],cwd=data.parent,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    url=f'http://127.0.0.1:{port}'
    try:
        for _ in range(150):
            try:
                with urllib.request.urlopen(url+'/api/session',timeout=.5):break
            except OSError:
                if process.poll() is not None:raise RuntimeError(process.stdout.read().decode())
                time.sleep(.1)
        else:raise RuntimeError('Synthetic provider QA server did not start')
        yield url
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()


@pytest.fixture
def mock_provider_page(browser,mock_provider_app_url,tmp_path):
    context=browser.new_context(viewport={'width':1440,'height':1000})
    page=context.new_page();page.set_default_timeout(15000)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(mock_provider_app_url)
    yield page
    page.screenshot(path=str(tmp_path/'provider-mock-final.png'),full_page=True)
    context.close()
    assert not errors, f'Browser errors: {errors}'
