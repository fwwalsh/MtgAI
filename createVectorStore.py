from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import JSONLoader
import json
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pprint import pprint

def main():
    
    chroma = Chroma(
        embedding_function=create_embeddings(), 
        collection_name="cards_collection",
        persist_directory="./data/chroma_db"
    )
    print("Loaded vector store")

    data = load_docs("cardsFiltered.json")
    print("Loaded data")

    batch_size = 5000  # Adjust based on your system's memory
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        chroma.add_documents(batch)
        print(f"Processed batch {i // batch_size + 1} of {len(data) // batch_size + 1}")

    print("Saved vector store")

def load_docs(json_file_path):
    """
    Loads JSON data from a file, extracts relevant text, and prepares it for embeddings.

    Args:
        json_file_path: The path to the JSON file.

    Returns:
        A list of cleaned text strings, suitable for embedding.
    """
    loader = JSONLoader(
        file_path=json_file_path,
        jq_schema='.[] | .document',
        text_content=False, # We will handle text content ourselves.
    )
    return loader.load()


def create_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def create_vector_store(chunks, embeddings):
    vectorStore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        collection_name="cards_collection",
        persist_directory="./data/chroma_db"
    )
    print("Saved vector store")
    return vectorStore

if __name__ == "__main__":
    main()