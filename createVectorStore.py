from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import JSONLoader
import json
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pprint import pprint

def load_and_prepare_json_for_embeddings(json_file_path):
    """
    Loads JSON data from a file, extracts relevant text, and prepares it for embeddings.

    Args:
        json_file_path: The path to the JSON file.

    Returns:
        A list of cleaned text strings, suitable for embedding.
    """
    loader = JSONLoader(
        file_path=json_file_path,
        jq_schema='.[].name', #act all values from each object
        text_content=False, # We will handle text content ourselves.
    )

    documents = loader.load()

    raw_texts = []
    for doc in documents:
        print(doc.page_content)
        raw_texts.append(str(doc.page_content)) # page_content is a dictionary, so convert to string.

    # Clean the raw text.
    import re
    cleaned_texts = []
    for text in raw_texts:
        text = re.sub(r"[^a-zA-Z0-9\s{}]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        cleaned_texts.append(text)

    return cleaned_texts

def load_cardsTxt():
    loader = TextLoader("cardsFiltered.Short.txt")
    documents = loader.load()
    print("Loaded documents")
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator="\n")
    chunks = text_splitter.split_documents(documents)
    return chunks

def print_cardsJson():
    with open("cardsFiltered.Short.json", "r") as f:
        cards = json.load(f)
        pprint(cards)
    return cards
#model = ChatVertexAI(model_name="gemini-2.0-flash-001", project="gen-lang-client-0391653967", key="AIzaSyAeToJ4AeUDvZHpa5ir705biWtbT_vbYVE")


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
data = load_and_prepare_json_for_embeddings("cardsFiltered.short.json")

pprint(data, indent=4, width=100, depth=4, compact=True)

print("Loaded cards")
