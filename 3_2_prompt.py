from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain, RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
import langchain

# langchain.debug = True

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
embeddings = OpenAIEmbeddings()
# create an instance of Chroma to perform only similarity search
db = Chroma(persist_directory="emb", embedding_function=embeddings)
retriever = db.as_retriever()

# chain type can be any of: ["stuff", "map_reduce", "map_rerank", "refine"]
chain = RetrievalQA.from_chain_type(llm=chat, retriever=retriever, chain_type="stuff")

result = chain.invoke("What is an interesting fact about the English Language?")  # type: ignore

print(result)
