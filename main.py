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


def create_rag(db):
    """
        Создание rag
    """

    # Настройки retriever
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Промпт
    template = """Используй следующий контекст для ответа на вопрос.
        Если ответ не найден в контексте, скажи "Я не знаю ответа на основе предоставленного контекста."

        Контекст:
        {context}

        Вопрос: {question}

        Ответ:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Запуск LLM через Ollama
    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.5,
        num_predict=512,
    )

    # Лямбда объединяет тексты из списка документов и передает в контекст/промпт LLM одним текстом с разделителями \n\n
    rag_from_docs = (
            RunnablePassthrough.assign(
                context=lambda x: "\n\n".join(doc.page_content for doc in x["context"])
            )
            | prompt
            | llm
            | StrOutputParser()
    )

    rag = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(result=rag_from_docs)

    print("\n" + "=" * 50)
    print("RAG с Ollama готов к труду и обороне")
    print("=" * 50 + "\n")

    return rag


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

    # Создаем два подключения к БД в главном потоке во избежание rust ошибок

    db_read_thread = database_manager.connect_database()
    if db_read_thread:
        print(f'[{get_current_time()}] Создано подключение к БД для рага: {db_read_thread}')

    db_write_thread = database_manager.connect_database()
    if db_write_thread:
        print(f'[{get_current_time()}] Создано подключение к БД для парсера: {db_write_thread}')

    # Отдельный тред для парсинга
    parser_thread = threading.Thread(
        target=parser.parsing_worker,
        args=(db_write_thread, 900),
        daemon=True
    )
    parser_thread.start()

    rag = create_rag(db_read_thread)

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
            source_docs = from_llm['context']

            # Если у модели нет ответа. Используем фразу отсутствия ответа из промпта
            no_answer_phrase = "Я не знаю ответа на основе предоставленного контекста"

            if no_answer_phrase in answer:
                print(
                    f"[{get_current_time()}] Милорд, в сообщениях шпиёнов ничегошеньки не сыскалось по сему вопросу...")
                continue

            # Нормальный ответ модели
            print(f"[{get_current_time()}] Ответствую, милорд. {answer}")

            # Источники ответа модели

            if source_docs:
                print(f"\nНамедни шпиёны сообщались нам:")
                for i, doc in enumerate(source_docs, 1):
                    print(f"\n{i}. {doc.page_content[:500]}.\n{doc.metadata.get('source', 'Неизвестный источник')}")


        except Exception as e:
            print(f"[{get_current_time()}] Милорд, толмач лыка не вяжет. Ошибочка вышла:\n {e}")


if __name__ == "__main__":
    main()
