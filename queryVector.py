import dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo


dotenv.load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables or gcloud ADC.")

# Card Retriever (Self-Query) - Schema remains the same
card_metadata_field_info = [
    AttributeInfo(name="name", description="The name of the Magic: The Gathering card", type="string"),
    AttributeInfo(name="type", description="The type line of the card (e.g., 'Artifact', 'Instant', 'Creature — Elf Druid')", type="string"),
    AttributeInfo(name="set", description="The Magic: The Gathering set the card is from", type="string"),
    AttributeInfo(name="power", description="The power of the creature card as an integer", type="int"),
    AttributeInfo(name="toughness", description="The toughness of the creature card", type="int"),
    AttributeInfo(name="mana_cost", description="The mana cost symbols of the card, not numeric", type="string"),
    AttributeInfo(name="loyalty", description="The loyalty of the planeswalker as an interger", type="int"),
    AttributeInfo(name="oracle_text", description="The Oracle text of the card, including abilities and effects", type="string"),
    AttributeInfo(name="isCommander", description="Whether the card is a commander card", type="boolean"),
    AttributeInfo(name="cmc", description="The converted mana cost of the card as an integer", type="int"),
]
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

vector_store = Chroma(
    persist_directory="./data/chroma_db",
    collection_name="cards_collection_google",
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), 
)
card_document_content_description = "Text content describing a Magic: The Gathering card, including its type, mana cost, set, abilities and stats."
card_retriever_prompt = ChatPromptTemplate.from_template("""Your goal is to structure the user's query to match the request schema provided below.

<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the following schema:

```json
{
    "query": string \ text string to compare to document contents
    "filter": string \ logical condition statement for filtering documents
}
```

The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.

A logical condition statement is composed of one or more comparison and logical operation statements.

A comparison statement takes the form: `comp(attr, val)`:
- `comp` (eq | ne | gt | gte | lt | lte): comparator
- `attr` (string):  name of attribute to apply the comparison to
- `val` (string): is the comparison value

A logical operation statement takes the form `op(statement1, statement2, ...)`:
- `op` (and | or): logical operator
- `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to

Make sure that you only use the comparators and logical operators listed above and no others.
Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters only use the attributed names with its function names if there are functions applied on them.
Make sure that filters only use format `YYYY-MM-DD` when handling date data typed values.
Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.
User Query: {question}
Structured Request:""")

card_retriever = SelfQueryRetriever.from_llm(
    llm,
    vector_store,
    card_document_content_description,
    card_metadata_field_info,
    verbose=True,
    search_kwargs={"k": 10},

    prompt=card_retriever_prompt
)

print(f"Vector store loaded. Collection '{vector_store._collection.name}' has approx {vector_store._collection.count()} items.")
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

template = """Answer the question based only on the following context.  Assume the user is searching for a commander card."
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": card_retriever | format_docs, "question": RunnablePassthrough() }
    | prompt
    | llm
    | StrOutputParser()
)

response = chain.invoke("Find an instant card that removes a creature from the battlefield with a cmc of 2 or less")
print("Response:", response)