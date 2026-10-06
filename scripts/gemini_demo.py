#!/usr/bin/env python3
"""Generated recording + real Gemini, in isolated storage with no response services."""
import argparse
import math
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, StrictBool
from starlette.middleware.trustedhost import TrustedHostMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vmd.config import load_environment
from vmd.gemini import IncidentAI
from vmd.media import MediaLibrary
from vmd.storage import Store


DEMO_ID = 'd'*32


def scene_frame(seconds):
    canvas = np.full((360, 640, 3), (29, 26, 24), dtype=np.uint8)
    cv2.rectangle(canvas, (25, 70), (615, 285), (55, 51, 47), -1)
    cv2.rectangle(canvas, (80, 86), (205, 195), (82, 72, 60), -1)
    cv2.rectangle(canvas, (435, 86), (560, 195), (82, 72, 60), -1)
    cv2.line(canvas, (25, 302), (615, 302), (125, 120, 108), 2)
    cv2.putText(canvas, 'GENERATED DEMO - NO REAL PEOPLE', (24, 32), cv2.FONT_HERSHEY_SIMPLEX, .61, (205, 210, 225), 1, cv2.LINE_AA)
    cv2.putText(canvas, f'Clip time {seconds:04.1f}s', (24, 338), cv2.FONT_HERSHEY_SIMPLEX, .5, (160, 165, 175), 1, cv2.LINE_AA)

    def person(x, color, side, angle=0, punching=False, walking=False):
        def point(local):
            a, b = local
            return (round(x+a*math.cos(angle)-b*math.sin(angle)), round(293+a*math.sin(angle)+b*math.cos(angle)))
        def line(a, b, tint=color, width=8):
            cv2.line(canvas, point(a), point(b), tint, width, cv2.LINE_AA)
        torso = np.array([point(v) for v in [(-15,-111),(15,-111),(14,-65),(-14,-65)]])
        cv2.fillConvexPoly(canvas, torso, color, cv2.LINE_AA)
        cv2.circle(canvas, point((0,-130)), 13, (190,190,190), -1, cv2.LINE_AA)
        line((0,-116),(0,-108),(190,190,190),6)
        stride=12*math.sin(seconds*8) if walking else 0
        for sign in (-1,1):
            line((sign*9,-65),(sign*11+stride*sign,-33),(170,170,170))
            line((sign*11+stride*sign,-33),(sign*18+stride*sign,0),(170,170,170))
            line((sign*18+stride*sign,0),(sign*18+stride*sign+8,0),(210,210,210),5)
        if punching:
            extension=(1+math.sin(seconds*math.pi*2.5+(0 if side==1 else math.pi)))/2
            line((side*17,-105),(side*(26+extension*25),-100-extension*9))
            line((side*(26+extension*25),-100-extension*9),(side*(32+extension*54),-89-extension*30))
            line((-side*17,-105),(-side*27,-88)); line((-side*27,-88),(-side*15,-117))
        else:
            for sign in (-1,1):
                line((sign*17,-105),(sign*24,-84)); line((sign*24,-84),(sign*20,-65))
    if seconds < 2:
        red=130+125*seconds/2; blue=505-155*seconds/2
    else:
        red=255 if seconds < 8 else 255-100*min(1,(seconds-8)/3)
        blue=350
    angle=math.pi/2*min(1,max(0,(seconds-6)/1.5))
    person(blue,(225,135,55),-1,angle,punching=2<=seconds<6,walking=seconds<2)
    person(red,(65,65,215),1,punching=2<=seconds<6,walking=seconds<2 or seconds>=8)
    return canvas


def seed_scene(store):
    if store.incident(DEMO_ID):
        return
    frames=[]
    for index in range(121):
        ok, image=cv2.imencode('.jpg',scene_frame(index/10),[cv2.IMWRITE_JPEG_QUALITY,85])
        if ok:
            frames.append((index/10,image.tobytes()))
    # live is the existing recording-analysis mode, never a physical camera.
    # This isolated app instantiates no alert, calling, SMS or evidence-sharing service.
    event={'id':DEMO_ID,'created':time.time(),'mode':'live','event_type':'possible_fight',
           'camera_id':'generated-recording','camera_name':'Generated animation - staged Gemini test',
           'score':.6,'reasons':['Test-only possible-fight label; no detector was run.'],
           'signals':{'live_camera':False,'location':None,'source_seconds':6,
                      'buffer_seconds':6,'generated_demo':True}}
    store._incident(event,frames)
    if not store.incident(DEMO_ID).get('clip'):
        raise RuntimeError('Could not encode the generated demo clip')


def create_demo_app(directory=None, run=False, provider=None):
    @asynccontextmanager
    async def lifespan(app):
        store=Store(directory or ROOT/'tmp/gemini-demo')
        seed_scene(store)
        ai=IncidentAI(store,provider=provider)
        app.state.store=store; app.state.ai=ai; app.state.media=MediaLibrary(store)
        if run:
            try: ai.start(DEMO_ID)
            except (RuntimeError,ValueError): pass
        yield
        ai.close();store.close()
    app=FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','testserver','[::1]'])
    @app.middleware('http')
    async def guard(request,call_next):
        if request.headers.get('origin') and request.headers['origin'] != f"{request.url.scheme}://{request.headers.get('host')}":
            return JSONResponse({'detail':'Cross-origin access denied'},status_code=403)
        if request.method != 'GET' and request.headers.get('x-vmd-client') != 'dashboard':
            return JSONResponse({'detail':'Local client header required'},status_code=403)
        response=await call_next(request)
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self'; frame-ancestors 'none'"
        return response
    @app.get('/')
    def index(): return FileResponse(ROOT/'vmd/static/gemini-demo.html')
    @app.get('/demo.mjs')
    def javascript(): return FileResponse(ROOT/'vmd/static/gemini-demo.mjs',media_type='text/javascript')
    @app.get('/demo.css')
    def css(): return FileResponse(ROOT/'vmd/static/gemini-demo.css',media_type='text/css')
    @app.get('/scene.mp4')
    def scene():
        clip=app.state.store.incident(DEMO_ID)['clip']
        return FileResponse(app.state.media.playback(app.state.store.clips/clip),media_type='video/mp4')
    @app.get('/api/status')
    def status():
        data=app.state.ai.status()
        return {key:data[key] for key in ['configured','backend','quality_model','fast_model','error']}
    @app.get('/api/report')
    def report(): return app.state.ai.report(DEMO_ID)
    class Analysis(BaseModel):
        quality: StrictBool=True
    class Question(BaseModel):
        question: str=Field(min_length=1,max_length=600)
    def action(fn,*args):
        try: return fn(*args)
        except (ValueError,RuntimeError) as error:
            raise HTTPException(503,str(error)) from None
    @app.post('/api/analyze')
    def analyze(body:Analysis): return action(app.state.ai.start,DEMO_ID,body.quality)
    @app.post('/api/ask')
    def ask(body:Question): return action(app.state.ai.ask,DEMO_ID,body.question)
    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Isolated generated scene + real Gemini demo, no calls or messaging')
    parser.add_argument('--port',type=int,default=8876)
    parser.add_argument('--run',action='store_true',help='Analyze the generated clip immediately using real Google inference')
    args=parser.parse_args()
    load_environment()
    # Only requested demo calls; never inherit automatic processing from main VDMA.
    os.environ['VMD_AI_AUTO_ANALYZE']='false'
    os.environ['VMD_AI_AUTO_FOLLOW']='false'
    import uvicorn
    uvicorn.run(create_demo_app(run=args.run),host='127.0.0.1',port=args.port,access_log=False)
