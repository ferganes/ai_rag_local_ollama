from config import *

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_ollama import OllamaLLM

import rag_engine.prompt as rag_engine


def create_rag(db):
    """
        Создание rag_engine
    """

    # Настройки retriever
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Промпт
    prompt = rag_engine.get_prompt()

    # Запуск LLM через Ollama
    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.5,
        num_predict=512,
    )

    # Лямбда объединяет тексты из чанков и передает в LLM одним текстом с разделителями \n\n
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
    print("RAG с Ollama запущены, милорд.")
    print("=" * 50 + "\n")

    return rag
