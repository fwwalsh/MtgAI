import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import JSONLoader
import json
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables or gcloud ADC.")

def main():
    
    chroma = Chroma(
        embedding_function=create_google_embeddings(), 
        collection_name="cards_collection_google",
        persist_directory="./data/chroma_db"
    )
    print("Loaded vector store")

    data = load_docs("./data/cardsFiltered.json")
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
    def metadata_func(record: dict, metadata: dict) -> dict:
        metadata_record = record.get("metadata")
        if metadata_record.get("name"):
            metadata["name"] = metadata_record.get("name")
        if metadata_record.get("type"):
            metadata["type"] = metadata_record.get("type")
        if metadata_record.get("set_name"):
            metadata["set_name"] = metadata_record.get("set_name")
        if metadata_record.get("loyalty") and metadata_record.get("loyalty").isdigit():
            metadata["loyalty"] = int(metadata_record.get("loyalty"))
        if (metadata_record.get("power") and metadata_record.get("power").isdigit()):
            metadata["power"] = int(metadata_record.get("power"))
        if (metadata_record.get("toughness") and metadata_record.get("toughness").isdigit()):
            metadata["toughness"] = int(metadata_record.get("toughness"))
        if metadata_record.get("oracle_text"):
            metadata["oracle_text"] = metadata_record.get("oracle_text")
        if metadata_record.get("mana_cost"):
            metadata["mana_cost"] = metadata_record.get("mana_cost")
        if metadata_record.get("cmc") :
            metadata["cmc"] = int(metadata_record.get("cmc"))
        if metadata_record.get("legalities").get("commanders"):
            metadata["isCommander"] = True
        else:
            metadata["isCommander"] = False    
        return metadata
    # don't forget legalities
    


    loader = JSONLoader(
        file_path=json_file_path,
        jq_schema='.[]',
        text_content=False, # We will handle text content ourselves.
        metadata_func=metadata_func, # Function to extract metadata from each record
        content_key="document", # The key in the JSON that contains the text content.
    )
    return loader.load()
#TODOL: code for google embeddings
def create_google_embeddings():
    #llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, convert_system_message_to_human=True)
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def create_huggingface_embeddings():
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