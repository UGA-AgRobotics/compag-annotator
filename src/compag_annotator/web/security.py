"""Loopback capability, same-origin mutations and request-size protection."""
from urllib.parse import urlsplit
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from compag_annotator.config import MAX_UPLOAD_BYTES

class LocalSecurity(BaseHTTPMiddleware):
    def __init__(self,app,token):super().__init__(app);self.token=token
    async def dispatch(self,request,call_next):
        host=request.headers.get('host','');parsed=urlsplit('http://'+host)
        if parsed.hostname not in ('127.0.0.1','localhost','::1','testserver'):return JSONResponse({'detail':'Local loopback access only'},status_code=403)
        origin=request.headers.get('origin')
        if origin and origin!=str(request.base_url).rstrip('/'):return JSONResponse({'detail':'Origin is not this local application'},status_code=403)
        if request.headers.get('sec-fetch-site')=='cross-site':return JSONResponse({'detail':'Cross-site request rejected'},status_code=403)
        if request.method not in ('GET','HEAD','OPTIONS') and request.headers.get('x-compag-token')!=self.token:return JSONResponse({'detail':'Local session token required'},status_code=403)
        try:size=int(request.headers.get('content-length','0'))
        except ValueError:return JSONResponse({'detail':'Invalid content length'},status_code=400)
        if size>MAX_UPLOAD_BYTES:return JSONResponse({'detail':'Request exceeds 256 MiB upload limit'},status_code=413)
        received=0
        original_receive=request._receive
        async def bounded_receive():
            nonlocal received
            message=await original_receive()
            if message['type']=='http.request':
                received+=len(message.get('body',b''))
                if received>MAX_UPLOAD_BYTES:
                    from starlette.exceptions import HTTPException
                    raise HTTPException(status_code=413,detail='Request exceeds upload limit')
            return message
        request._receive=bounded_receive
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff';response.headers['Referrer-Policy']='no-referrer'
        response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; form-action 'self'"
        response.headers['Cache-Control']='no-store'
        return response
