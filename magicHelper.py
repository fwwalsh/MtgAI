import json
import os
from dotenv import load_dotenv
import shutil # To clean up Chroma persistence if needed
from pprint import pprint

# --- LangChain Core Components ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever # Still useful for query rewriting
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Document Loaders ---
from langchain_community.document_loaders import JSONLoader, PyPDFLoader # Added PyPDFLoader
from langchain.docstore.document import Document

# --- Text Splitters ---
from langchain.text_splitter import RecursiveCharacterTextSplitter # Added for splitting PDF pages

# --- Embeddings ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Vector Stores ---
from langchain_community.vectorstores import Chroma

# --- Chat Models ---
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Retrievers ---
from langchain.retrievers import ContextualCompressionRetriever # Optional
from langchain.retrievers.document_compressors import LLMChainExtractor # Optional
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# --- Memory ---
from langchain_community.chat_message_histories import ChatMessageHistory

# --- 0. Setup ---
load_dotenv()

# Ensure Google API key is available
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables or gcloud ADC.")

# --- 3. Initialize Gemini LLM & Embeddings (Same as before) ---
print("Initializing Gemini models...")



# --- 5. Define Metadata Schema & Create Retrievers (Same logic) ---

# Card Retriever (Self-Query) - Schema remains the same
card_metadata_field_info = [
    AttributeInfo(name="name", description="The name of the Magic: The Gathering card", type="string"),
    AttributeInfo(name="type", description="The type line of the card (e.g., 'Artifact', 'Creature — Elf Druid')", type="string"),
    AttributeInfo(name="set", description="The Magic: The Gathering set the card is from", type="string"),
    AttributeInfo(name="power", description="The power of the creature card", type="string"),
    AttributeInfo(name="toughness", description="The toughness of the creature card", type="string"),
    AttributeInfo(name="mana_cost", description="The mana cost symbols of the card", type="string"),
]
card_document_content_description = "Text content describing a Magic: The Gathering card, including its abilities and stats."

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, convert_system_message_to_human=True)
cards_vectorstore = Chroma(
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), 
        collection_name="cards_collection_google",
        persist_directory="./data/chroma_db"
    )

card_retriever_base = SelfQueryRetriever.from_llm(
    llm,
    cards_vectorstore,
    card_document_content_description,
    card_metadata_field_info,
    verbose=True
)
print("Card Self-query retriever created.")

rules_vectorstore = Chroma(
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), 
        collection_name="mtg_comprehensive_rules",
        persist_directory="./data/chroma_db"
    )


# Rules Retriever (Basic Vector Store Retriever for PDF chunks)
# Metadata filtering on rules is less likely needed/useful here,
# as the primary metadata is just 'page' number.
rules_retriever_base = rules_vectorstore.as_retriever(search_kwargs={"k": 5}) # Increase k slightly due to chunking
print("Rules basic retriever created (from PDF chunks).")

# Assign retrievers (without compression for simplicity)
card_retriever = card_retriever_base
rules_retriever = rules_retriever_base

# --- 6. Build Combined RAG Chain (Same logic as before) ---

# 6a. History-Aware Query Rewriting Chain
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [ ("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}"), ]
)


#history_aware_retriever_chain = create_history_aware_retriever(
#        llm, RunnableParallel(cards=card_retriever, rules=rules_retriever), contextualize_q_prompt
#    ) | RunnableLambda(lambda x: x['query'])



# 6b. Combined Retrieval Function
def get_combined_documents(query: str):
    """Fetches documents from both card and rule retrievers and combines them."""
    print(f"\n--- Retrieving documents for query: '{query}' ---")
    card_docs = card_retriever.invoke(query)
    print(f"Retrieved {len(card_docs)} card documents.")
    rule_docs = rules_retriever.invoke(query)
    print(f"Retrieved {len(rule_docs)} rule documents (from PDF chunks).")
    # Add source info
    for doc in card_docs: doc.metadata["source_type"] = "card_database"
    for doc in rule_docs: doc.metadata["source_type"] = "rules_pdf" # Use different key/value
    combined = card_docs + rule_docs
    print(f"Total combined documents: {len(combined)}")
    return combined

# 6c. Document Combination and Answer Generation Chain
qa_system_prompt = """You are an assistant for question-answering tasks about Magic: The Gathering cards and rules. \
Use the following pieces of retrieved context, which may come from card data ('card_database') or the official rules PDF ('rules_pdf'), to answer the question. \
If context from the rules PDF includes a page number, you can mention it if helpful (e.g., 'mentioned on page X of the rules PDF'). \
If you don't know the answer based on the context, just say that you don't know. \
Keep the answer concise and accurate.

Context:
{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [ ("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}"), ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# 6d. Full RAG Chain Assembly using LCEL
def route_input_for_retrieval(inputs):
    
    standalone_query = history_aware_retriever_chain.invoke(inputs)
    combined_docs = get_combined_documents(standalone_query)
    return { "context": combined_docs, "input": inputs["input"], "chat_history": inputs["chat_history"] }

rag_chain = ( RunnableLambda(route_input_for_retrieval) | question_answer_chain )
print("Full RAG chain with dual retrieval (PDF rules) created.")

# --- 7. Add Memory (Same as before) ---
store = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store: store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain, get_session_history,
    input_messages_key="input", history_messages_key="chat_history",
    output_messages_key="answer",
)
print("Conversational RAG chain with history created.")

# --- 8. Run Conversational Interaction (Same queries) ---
session_id_1 = "mtg_gemini_dual_pdf_chat_1"
print("\n--- Conversation Start (Session ID: {}) ---".format(session_id_1))

def ask_question(query, session_id):
    print(f"\nUser Query: {query}")
    response = conversational_rag_chain.invoke(
        {"input": query}, config={"configurable": {"session_id": session_id}}
    )
    print("\nAssistant Response:")
    pprint(response['answer'])

# Query 1: Card specific
ask_question("What does Static Orb do?", session_id_1)

# Query 2: Rule specific (Should now hit the PDF content)
ask_question("What is the golden rule about card text vs rules?", session_id_1)

# Query 3: Follow-up relying on history (Static Orb)
ask_question("Does it involve tapping?", session_id_1)

# Query 4: Combining card and rule (Deathtouch)
ask_question("Tell me about deathtouch. Are there any cards with it in the database?", session_id_1)

# Query 5: Metadata filter (Creatures from Dominaria United)
ask_question("Show me creatures from Dominaria United.", session_id_1)

# Query 6: Rule detail (Should hit PDF content)
ask_question("What does the rules PDF say about flying?", session_id_1)

print("\n--- Retrieving Chat History for Session '{}' ---".format(session_id_1))
history = get_session_history(session_id_1)
printable_history = []
for msg in history.messages:
    if isinstance(msg, BaseMessage): printable_history.append({"type": msg.type, "content": msg.content})
    else: printable_history.append(str(msg))
pprint(printable_history)

# --- Optional: Clean up Chroma directories ---
# shutil.rmtree(cards_chroma_dir)
# shutil.rmtree(rules_chroma_dir)
# print("\nCleaned up Chroma directories.")
# --------------------------------------------
