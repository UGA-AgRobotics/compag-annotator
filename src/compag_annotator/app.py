"""Installed local application; importing it never imports a machine-learning stack."""
from contextlib import asynccontextmanager
from pathlib import Path
import secrets
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from compag_annotator import __version__
from compag_annotator.config import application_data
from compag_annotator.core.projects import ConflictError
from compag_annotator.web.service import Service
from compag_annotator.web.security import LocalSecurity
from compag_annotator.web.api import router

def create_app(data_dir=None,token=None,qa_mode=False):
    service=Service(application_data(data_dir),qa_mode);token=token or secrets.token_urlsafe(32)
    @asynccontextmanager
    async def lifespan(app):
        yield
        service.jobs.shutdown()
        if service._manager is not None:service.manager.unload()
    app=FastAPI(title='COMPAG Annotator',version=__version__,lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
    app.state.service=service;app.state.token=token
    app.add_middleware(LocalSecurity,token=token)
    @app.exception_handler(ConflictError)
    async def conflict(request,error):return JSONResponse({'detail':str(error)},status_code=409)
    @app.exception_handler(ValueError)
    async def invalid(request,error):return JSONResponse({'detail':str(error),**({'loss_report':error.loss_report} if hasattr(error,'loss_report') else {}),**({'fields':error.fields} if hasattr(error,'fields') else {})},status_code=400)
    @app.exception_handler(KeyError)
    async def missing(request,error):return JSONResponse({'detail':'Missing required field: '+str(error)},status_code=400)
    @app.exception_handler(OSError)
    async def storage_error(request,error):return JSONResponse({'detail':'Local file operation failed: '+str(error)},status_code=400)
    app.include_router(router(service,token))
    assets=Path(__file__).parent/'web/assets'
    app.mount('/assets',StaticFiles(directory=assets,check_dir=False),name='assets')
    app.mount('/help',StaticFiles(directory=Path(__file__).parent/'help',html=True,check_dir=False),name='help')
    @app.get('/')
    def home():return FileResponse(assets/'index.html')
    return app
