from config import *
import threading


from utils.current_time import get_current_time
from utils.threading_event import input_active

import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import parser.parser as parser
import database_manager.database_manager as database_manager
import rag_engine.rag as rag_engine


# Проверка доступности локальной Ollama
def check_ollama():
    try:
        # Проверяем, запущена ли Ollama
        response = requests.get("http://localhost:11434", timeout=5)

        if response.status_code == 200:
            print(f"OK | Ollama найдена и работает с LLM {LLM_MODEL} и embedding {EMBEDDING_MODEL}.")
            return True

    except:
        print("X | Ollama недоступна...")
        return False


def main():

    print(f'Время запуска {get_current_time()}')

    # Проверяем доступность локальной ollama
    if not check_ollama():
        return

    # Удаление БД
    # database_manager.drop_database()

    # Проверяем БД. Автоматически создаем новую, если БД нет
    if not database_manager.check_database_exists():
        database_manager.create_database()

    db_read_thread = database_manager.connect_database()
    if db_read_thread:
        print(f'[{get_current_time()}] Создано подключение к БД для раг')

    db_write_thread = database_manager.connect_database()
    if db_write_thread:
        print(f'[{get_current_time()}] Создано подключение к БД для парсера')

    parser_thread = threading.Thread(
        target=parser.parsing_worker,
        args=(db_write_thread, 900),
        daemon=True
    )

    parser_thread.start()

    rag = rag_engine.create_rag(db_read_thread)

    # Цикл вопрос-ответ
    while True:
        input_active.set()
        question = input(f"\n[{get_current_time()}] Чего изволите, милорд?! (или 'exit' для выхода): ").strip()
        input_active.clear()

        if question.lower() == 'exit':
            print(f"[{get_current_time()}] Как угодно, милорд. За дверями буду...")
            break

        if not question:
            print(f"[{get_current_time()}] Милорд?!")
            continue

        # Ответ LLM
        try:
            from_llm = rag.invoke(question)

            answer = from_llm['result']
            source = from_llm['context']

            # Ответ модели
            print(f"[{get_current_time()}] Ответствую, милорд. {answer}")

            # Источники ответа модели
            if source:
                for i, doc in enumerate(source, 1):
                    print(f"\n{i}. [...{doc.page_content[:500]}...]\n{doc.metadata.get('source', 'Неизвестный источник')}")

        except Exception as e:
            print(f"[{get_current_time()}] Милорд, толмач лыка не вяжет. Ошибочка вышла:\n {e}")


if __name__ == "__main__":
    main()
