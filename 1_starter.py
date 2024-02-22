import os
from pprint import pprint
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain


load_dotenv()

# llm = ChatOpenAI(model="gpt-4-0125-preview")
llm = ChatOpenAI()
# output_parser = StrOutputParser()
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are world class technical documentation writer."),
#         ("user", "{input}"),
#     ]
# )

# chain = prompt | llm | output_parser

# print(chain.invoke({"input": "how can langsmith help with testing?"}))


code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"],
)

test_prompt = PromptTemplate(
    template="Write a test for the following {language} code:\n{code}",
    input_variables=["language", "code"],
)

code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="code")
test_chain = LLMChain(llm=llm, prompt=test_prompt, output_key="test")

# chain = SequentialChain(
#     chains=[code_chain, test_chain],
#     input_variables=["task", "language"],
#     output_variables=["test", "code"],
# )
chain = code_chain | test_chain

result = chain.invoke({"language": "python", "task": "return a list of 10 numbers"})

# print(result["text"])
# pprint(result, indent=2)
print("\n>>>>> Generated code:")
print(result["code"])
print("\n>>>>> Generated test:")
print(result["test"])
