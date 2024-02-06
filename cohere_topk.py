# get a new token: https://dashboard.cohere.ai/

import getpass
import os

os.environ["COHERE_API_KEY"] = getpass.getpass("Cohere API Key:")

# Helper function for printing docs


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS

documents = TextLoader("text.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
retriever = FAISS.from_documents(texts, CohereEmbeddings()).as_retriever(
    search_kwargs={"k": 20}
)

query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.get_relevant_documents(query)
pretty_print_docs(docs)