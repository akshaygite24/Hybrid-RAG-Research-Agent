from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


def get_retriever(search_k: int=4):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": search_k}
    )

def retrieve_docs(query: str, k: int = 4):
    retriever = get_retriever(k)
    docs = retriever.invoke(query)
    return docs

if __name__=="__main__":
    results = retrieve_docs("Explain AI?")
    for i, doc in enumerate(results):
        print(f"\n---Chunk {i+1} ---")
        print(f"Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}")
        print(doc.page_content[:300])