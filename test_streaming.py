import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


model = ChatOpenAI(model="gpt-4-0125-preview")
model = ChatOpenAI()

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
parser = StrOutputParser()
chain = prompt | model | parser


# Wrap the async for loop in an asynchronous function
async def get_joke():
    async for chunk in chain.astream({"topic": "parrot"}):
        print(chunk, flush=True)


# Function to run the async get_joke function
def run():
    asyncio.run(get_joke())


run()
