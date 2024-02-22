from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel
from langchain.tools import StructuredTool
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
            content="You are an AI that has access to a SQLite database.\n"
            f"The database has tables of: {tables}\n"
            "Do not make any assumption about what table exist or what column exist. Instead, use the 'describe_tables' function."
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
)


def write_report(filename, html):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)


class WriteReportArgsSchema(BaseModel):
    filename: str
    html: str


# Tool can only use function that receive a single argument. We need to use StructuredTool
write_report_tool = StructuredTool.from_function(
    name="write_report",
    description="Write an HTML to disk. Use this tool whenever someone asks for a report.",
    func=write_report,
    args_schema=WriteReportArgsSchema,
)

tools = [run_query_tool, describe_tables_tool, write_report_tool]

agent = create_openai_functions_agent(llm=chat, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_executor("How many users are in the database?")
agent_executor.invoke(
    {
        "input": "Summarize the top most 5 popular products. Write the results to a report file"
    }
)
