from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_ollama import OllamaLLM

from config import *


def create_rag_agent(rag):

    """ReAct агент"""

    @tool
    def rag_search(query: str) -> str | None:
        """Поиск по статьям в локальной базе через раг"""

        try:
            result = rag.invoke(query)
            return result["result"]

        except Exception:
            return None

    tools = [rag_search]

    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.5,
        num_predict=1024,
    )

    prompt = PromptTemplate.from_template(
        """Ты - аналитик политических новостей с доступом к базе знаний.

        У тебя есть доступ к следующим инструментам:
        {tools}

        Используй следующий формат:

        Question: вопрос пользователя
        Thought: подумай, что нужно сделать
        Action: название инструмента (должно быть одним из: {tool_names})
        Action Input: входные данные для инструмента
        Observation: результат работы инструмента
        ... (этот цикл Thought/Action/Action Input/Observation может повторяться)
        Thought: теперь я знаю ответ
        Final Answer: твой ответ пользователю

        Важно: 
        - Для поиска информации используй инструмент rag_search
        - В Final Answer давай ответ в формате:
          Ответ: [конкретный факт]
          Контекст/причины: [объяснение]

        Начни!

        Question: {input}
        {agent_scratchpad}"""
    )

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=2
    )

    return agent_executor
