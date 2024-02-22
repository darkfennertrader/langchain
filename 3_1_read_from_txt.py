# %%
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter


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


embeddings = OpenAIEmbeddings()
# emb = embeddings.embed_query(("Hi there"))

# %%

# chunk_size has priority over separator
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=0)
loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)
# text -> embeddings -> vectors storing
# (every time it runs it makes duplication of the embeddings)
db = Chroma.from_documents(docs, embedding=embeddings, persist_directory="emb")

# for doc in docs:
#     print(doc.page_content, "\n")

results = db.similarity_search(
    "What is an interesting fact about the English language ?", k=4
)


for result in results:
    print()
    print(result.page_content, "\n")
