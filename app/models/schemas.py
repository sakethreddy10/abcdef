from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., description='Natural language question to query against the documents.', examples=['What is the annual leave policy?'])
    bucket: str | None = Field(default=None, description="Target bucket to query ('bucket_1' or 'bucket_2'). If omitted, searches across all buckets.", examples=['bucket_1'])
    top_k: int = Field(default=5, ge=1, le=20, description='Number of most relevant context chunks to retrieve.')

class SourceCitation(BaseModel):
    document: str = Field(..., description='Filename of the source document.')
    page: str | int | None = Field(default=None, description='Page or sheet number where the fact was found.')
    bucket: str | None = Field(default=None, description='Bucket the source document belongs to.')
    file_type: str | None = Field(default=None, description='File extension/type of the document.')

class QueryResponse(BaseModel):
    question: str
    answer: str
    bucket: str | None = None
    sources: list[SourceCitation] = Field(default_factory=list)

class IngestRequest(BaseModel):
    bucket: str = Field(default='bucket_1', description="Bucket identifier for the documents being ingested ('bucket_1' or 'bucket_2').", examples=['bucket_1'])
    directory_path: str | None = Field(default=None, description='Optional custom path to ingest documents from. If omitted, uses default bucket path from config.')

class IngestResponse(BaseModel):
    status: str
    bucket: str
    documents_parsed: int
    chunks_stored: int
    message: str

class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    ollama_model: str
    vector_store_chunks: int
    embedding_model: str

class DocumentsStatsResponse(BaseModel):
    total_chunks: int
    collection_name: str
    embedding_dimension: int
