from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import BaseRetriever
import langchain

langchain.debug = True


load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()
# create an instance of Chroma to perform only similarity search
db = Chroma(persist_directory="emb", embedding_function=embeddings)


class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query):
        # calculate emebddings for the query strings
        emb = self.embeddings.embed_query(query)

        # take embeddings and feed them into that
        # max_marginal_relevance_search_by_vector

        # lambda_mult control how similar the documents returned can be. Lower the value to get more diverse documents
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb, lambda_mult=0.8
        )

    def aget_relevant_documents(self, query):
        return []


# this retriever eliminates duplicates
retriever = RedundantFilterRetriever(embeddings=embeddings, chroma=db)

# the plain retriever does not eliminate duplicates
# retriever = db.as_retriever()

# chain type can be any of: ["stuff", "map_reduce", "map_rerank", "refine"]
chain = RetrievalQA.from_chain_type(llm=chat, retriever=retriever, chain_type="stuff")

result = chain.invoke("What is an interesting fact about the English Language?")  # type: ignore

print(result)
