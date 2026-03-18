import os
import shutil
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "qwen2.5:14b"  # qwen2.5:14b / qwen3.5 /

persist_dir = "./chroma_db"

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)

collection_name = "article_rss_lenta_ru"


def check_database_exists():
    """
    Проверяет существование директории и файла базы данных.
    Возвращает True, если всё ок. False, если всё не ок.
    """
    if not os.path.exists(persist_dir):
        return False

    db_file = os.path.join(persist_dir, "chroma.sqlite3")
    return os.path.exists(db_file)


def create_database():
    """
    Создаёт директорию для новой БД.
    Возвращает True, если всё ок. False, если всё не ок.
    """
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
        print(f"--> Создана директория базы данных: {persist_dir}")
        return True
    return False


def connect_database() -> Chroma:
    """
    Подключается к БД или создаёт новую автоматически.

    """
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name
    )

    return db


def update_database(db, docs):
    """
    Добавляет документы в базу данных

    Args:
        db: ChromaDB
        docs: список Document
    """

    db.add_documents(docs)


def drop_database():
    shutil.rmtree(persist_dir)
    print(f"--> БД в директории {persist_dir} удалена...")


