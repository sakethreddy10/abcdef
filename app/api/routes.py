import os
import logging
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
import json
from app.models.schemas import QueryRequest, QueryResponse, IngestRequest, IngestResponse, HealthResponse, DocumentsStatsResponse, SourceCitation
from app.config import BUCKET_1_PATH, BUCKET_2_PATH, OLLAMA_MODEL, EMBEDDING_MODEL
from app.services.ingestion_service import ingest_directory, ingest_directory_incremental, file_fingerprint, ingest_file, PARSER_MAP
from app.services.chunking_service import chunk_documents
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
logger = logging.getLogger(__name__)
router = APIRouter()
SUPPORTED_MIME_TYPES = {'.pdf': {'application/pdf'}, '.docx': {'application/vnd.openxmlformats-officedocument.wordprocessingml.document'}, '.txt': {'text/plain', 'application/octet-stream'}, '.csv': {'text/csv', 'application/vnd.ms-excel', 'application/octet-stream'}, '.xlsx': {'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}, '.xls': {'application/vnd.ms-excel'}, '.png': {'image/png'}, '.jpg': {'image/jpeg'}, '.jpeg': {'image/jpeg'}}
embedder = EmbeddingService()
vector_store = VectorStoreService()
retriever = RetrievalService(embedder, vector_store)
llm = LLMService()
rag_service = RAGService(retriever, llm)

@router.get('/health', response_model=HealthResponse, summary='System Health Check', tags=['System'])
def health_check():
    ollama_ok = llm.is_available()
    doc_count = vector_store.count()
    return HealthResponse(status='healthy' if ollama_ok else 'degraded', ollama_connected=ollama_ok, ollama_model=OLLAMA_MODEL, vector_store_chunks=doc_count, embedding_model=EMBEDDING_MODEL)

@router.get('/documents', response_model=DocumentsStatsResponse, summary='Vector Store Statistics', tags=['Documents'])
def get_documents_stats():
    return DocumentsStatsResponse(total_chunks=vector_store.count(), collection_name=vector_store._collection.name, embedding_dimension=embedder._model.get_embedding_dimension())

@router.post('/ingest', response_model=IngestResponse, status_code=status.HTTP_201_CREATED, summary='Ingest Documents from Bucket', tags=['Ingestion'])
def ingest_documents(payload: IngestRequest):
    bucket = payload.bucket
    if bucket not in ['bucket_1', 'bucket_2']:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid bucket. Must be 'bucket_1' or 'bucket_2'.")
    target_dir = payload.directory_path
    if not target_dir:
        target_dir = BUCKET_1_PATH if bucket == 'bucket_1' else BUCKET_2_PATH
    target_dir = os.path.abspath(target_dir)
    if not os.path.exists(target_dir):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'Bucket directory does not exist: {target_dir}')
    try:
        docs, changed_sources, removed_sources = ingest_directory_incremental(target_dir, bucket, vector_store)
        vector_store.delete_sources(bucket, changed_sources + list(removed_sources))
        if not docs:
            return IngestResponse(status='success', bucket=bucket, documents_parsed=0, chunks_stored=0, message=f'No parseable documents found in {target_dir}')
        chunks = chunk_documents(docs)
        document_hashes = {document['source']: document.get('document_hash', '') for document in docs}
        for chunk in chunks:
            chunk['document_hash'] = document_hashes.get(chunk['source'], '')
        texts = [c['text'] for c in chunks]
        embeddings = embedder.embed(texts)
        stored_count = vector_store.add_chunks(chunks, embeddings)
        return IngestResponse(status='success', bucket=bucket, documents_parsed=len(docs), chunks_stored=stored_count, message=f'Successfully ingested {len(docs)} document item(s) producing {stored_count} chunk(s) into {bucket}.')
    except Exception as e:
        logger.error(f'Ingestion failed: {e}', exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Ingestion failed: {str(e)}')

@router.post('/upload/{bucket}', response_model=IngestResponse, status_code=status.HTTP_201_CREATED, summary='Upload and Index Documents', tags=['Ingestion'])
async def upload_documents(bucket: str, files: list[UploadFile]=File(...)):
    bucket_paths = {'bucket_1': BUCKET_1_PATH, 'bucket_2': BUCKET_2_PATH}
    if bucket not in bucket_paths:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid bucket. Must be 'bucket_1' or 'bucket_2'.")
    target_dir = os.path.abspath(bucket_paths[bucket])
    os.makedirs(target_dir, exist_ok=True)
    saved_names = []
    parsed_documents = []
    try:
        for upload in files:
            filename = os.path.basename(upload.filename or '')
            extension = os.path.splitext(filename)[1].lower()
            if not filename or extension not in PARSER_MAP:
                supported = ', '.join(PARSER_MAP.keys())
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported or missing filename '{filename}'. Supported types: {supported}")
            mime_type = (upload.content_type or '').split(';', 1)[0].strip().lower()
            if mime_type and mime_type not in SUPPORTED_MIME_TYPES[extension]:
                expected = ', '.join(sorted(SUPPORTED_MIME_TYPES[extension]))
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"MIME type mismatch for '{filename}': received '{mime_type}', expected one of: {expected}.")
            destination = os.path.join(target_dir, filename)
            with open(destination, 'wb') as output_file:
                while (content := (await upload.read(1024 * 1024))):
                    output_file.write(content)
            saved_names.append(filename)
            fingerprint = file_fingerprint(destination)
            if vector_store.source_fingerprints(bucket).get(filename) == fingerprint:
                logger.info(f'Skipping unchanged file: {filename}')
                continue
            parsed = ingest_file(destination, bucket)
            for document in parsed:
                document['document_hash'] = fingerprint
            parsed_documents.extend(parsed)
        if not parsed_documents:
            return IngestResponse(status='success', bucket=bucket, documents_parsed=0, chunks_stored=0, message=f"Uploaded {', '.join(saved_names)}, but no readable text was extracted.")
        chunks = chunk_documents(parsed_documents)
        document_hashes = {document['source']: document.get('document_hash', '') for document in parsed_documents}
        for chunk in chunks:
            chunk['document_hash'] = document_hashes.get(chunk['source'], '')
        embeddings = embedder.embed([chunk['text'] for chunk in chunks])
        vector_store.delete_sources(bucket, [source for source in saved_names if source in {document['source'] for document in parsed_documents}])
        stored_count = vector_store.add_chunks(chunks, embeddings)
        return IngestResponse(status='success', bucket=bucket, documents_parsed=len(parsed_documents), chunks_stored=stored_count, message=f"Uploaded and indexed {', '.join(saved_names)} ({stored_count} chunk(s)).")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Upload failed: {e}', exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Upload failed: {str(e)}')

@router.get('/documents/sources', tags=['Documents'])
def get_indexed_sources():
    result = vector_store._collection.get(include=['metadatas'])
    sources = {}
    for metadata in result.get('metadatas', []):
        key = (metadata.get('bucket'), metadata.get('source'))
        sources[key] = {'document': metadata.get('source'), 'bucket': metadata.get('bucket'), 'file_type': metadata.get('file_type')}
    return {'sources': list(sources.values())}

@router.get('/documents/chunks', tags=['Documents'])
def get_document_chunks(source: str | None=Query(default=None, description='Optional exact indexed document filename.'), bucket: str | None=Query(default=None, description='Optional bucket filter.'), limit: int=Query(default=20, ge=1, le=100), offset: int=Query(default=0, ge=0), include_embeddings: bool=Query(default=False)):
    if bucket is not None and bucket not in ['bucket_1', 'bucket_2']:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid bucket. Must be 'bucket_1' or 'bucket_2'.")
    result = vector_store.inspect_chunks(limit=limit, offset=offset, bucket=bucket, source=source, include_embeddings=include_embeddings)
    if result['returned_chunks'] == 0 and source:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No indexed chunks found for source '{source}'.")
    return result

@router.post('/query', response_model=QueryResponse, status_code=status.HTTP_200_OK, summary='Query Documents (RAG Pipeline)', tags=['Query'])
def query_rag(payload: QueryRequest):
    if vector_store.count() == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='The vector store is empty. Please run /ingest first before querying.')
    try:
        result = rag_service.query(question=payload.question, bucket=payload.bucket, top_k=payload.top_k)
        sources = [SourceCitation(document=s.get('document', 'Unknown'), page=s.get('page'), bucket=s.get('bucket'), file_type=s.get('file_type')) for s in result.get('sources', [])]
        return QueryResponse(question=result['question'], answer=result['answer'], bucket=result['bucket'], sources=sources)
    except ConnectionError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f'Ollama server is unavailable: {str(e)}')
    except Exception as e:
        logger.error(f'RAG query execution failed: {e}', exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Error executing RAG query: {str(e)}')

@router.post('/query/stream', summary='Stream Query Answer (RAG Pipeline)', tags=['Query'])
def stream_query_rag(payload: QueryRequest):
    try:
        is_empty = vector_store.count() == 0
    except Exception as e:
        logger.error(f'Unable to inspect vector store before streaming: {e}', exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Unable to access the vector store: {e}')
    if is_empty:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='The vector store is empty. Please run /ingest first before querying.')

    def event_stream():
        try:
            yield (json.dumps({'type': 'status', 'content': 'Searching indexed context...'}) + '\n')
            for event in rag_service.stream_query(question=payload.question, bucket=payload.bucket, top_k=payload.top_k):
                yield (json.dumps(event) + '\n')
        except ConnectionError as e:
            yield (json.dumps({'type': 'error', 'detail': f'Ollama server is unavailable: {e}'}) + '\n')
        except Exception as e:
            logger.error(f'Streaming RAG query failed: {e}', exc_info=True)
            yield (json.dumps({'type': 'error', 'detail': f'Error executing RAG query: {e}'}) + '\n')
    return StreamingResponse(event_stream(), media_type='application/x-ndjson', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
