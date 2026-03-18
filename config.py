from langchain_ollama import OllamaEmbeddings

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "qwen2.5:14b"  # qwen2.5:14b / qwen3.5 /

URL_FEED = 'https://lenta.ru/rss/news/world'

persist_dir = "./chroma_db"

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)

collection_name = "article_rss_lenta_ru"

