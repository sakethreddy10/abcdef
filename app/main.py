import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from app.api.routes import router
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/app.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
logging.getLogger().addHandler(file_handler)
logger = logging.getLogger('app.main')
app = FastAPI(title='Local Document RAG API', description='Enterprise-grade local Document RAG Pipeline using Ollama, ChromaDB, and SentenceTransformers with 2-bucket document isolation.', version='1.0.0', docs_url='/docs', redoc_url='/redoc')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.include_router(router, prefix='')

@app.get('/', tags=['Root'])
def root():
    return FileResponse('app/static/index.html', headers={'Cache-Control': 'no-store'})

@app.get('/favicon.ico', include_in_schema=False)
def favicon():
    return Response(status_code=204)
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app.main:app', host='0.0.0.0', port=8000, reload=True)

# Code update
