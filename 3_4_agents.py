from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import BaseRetriever
from langchain.agents import (
    AgentExecutor,
    create_openai_functions_agent,
)
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from tools.sql import run_query_tool, list_tables, describe_tables_tool


load_dotenv()

# import langchain
# chat = ChatOpenAI(model="gpt-4-0125-preview")
chat = ChatOpenAI()

tables = list_tables()

prompt = ChatPromptTemplate(
    input_variables=[],
    messages=[
        SystemMessage(
            content=f"You are an AI that has access to a SQLite database.\n"
            # f"The database has tables of: {tables}\n"
            # "Do not make any assumption about what table exist "
            # "or what column exist. Instead, use the 'describe_tables' function."
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
)
tools = [run_query_tool, describe_tables_tool]

agent = create_openai_functions_agent(llm=chat, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_executor("How many users are in the database?")
agent_executor.invoke({"input": "How many users have a shipping address"})
