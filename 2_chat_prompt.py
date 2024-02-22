from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import (
    ConversationBufferMemory,
    FileChatMessageHistory,
    ConversationSummaryMemory,
)

load_dotenv()

chat = ChatOpenAI()

# OPTION 1: builds up memory by accumulating all the history of messages in a file.json
# memory = ConversationBufferMemory(
#     chat_memory=FileChatMessageHistory("messages.json"),
#     memory_key="messages",
#     return_messages=True,
# )

# OPTION 2: builds up memory by summarizing all the previous messages using chat as an LLM
memory = ConversationSummaryMemory(
    memory_key="messages", return_messages=True, llm=chat
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

# verbose=True for debugging
chain = LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)


while True:
    content = input(">> ")
    result = chain.invoke({"content": content})
    print(result["text"])
