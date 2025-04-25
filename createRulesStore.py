from langchain_google_genai import GoogleGenerativeAIEmbeddings
import requests
import re
import chromadb
import os
from chromadb.utils import embedding_functions

# --- Configuration ---
# NOTE: The URL for the comprehensive rules TXT file might change.
# Check the official Magic: The Gathering website or resources like MTG Wiki
# (https://mtg.fandom.com/wiki/Comprehensive_Rules) for the latest link.
# Update this URL if the script fails to download the file.
RULES_URL = "https://media.wizards.com/docs/magicdocs/MagicCompRules_20240415.txt" # Example URL - UPDATE AS NEEDED
LOCAL_RULES_FILE = "./data/cardRules.txt"
CHROMA_DB_PATH = "./data/chroma_db" # Path to store the ChromaDB database
CHROMA_COLLECTION_NAME = "mtg_comprehensive_rules"



def parse_rules(file_path: str) -> list[dict[str, str]]:
    """
    Parses the downloaded rules text file into individual rule sections.

    Args:
        file_path: Path to the local rules text file.

    Returns:
        A list of dictionaries, where each dictionary represents a rule
        and has 'id' (rule number) and 'text' (rule content).
        Returns an empty list if parsing fails or the file doesn't exist.
    """
    rules_data = []
    current_rule_id = None
    current_rule_text = ""

    # Regex to identify the start of a rule (e.g., "100.", "100.1.", "702.15b.")
    # It looks for digits, optional dot, optional digits, optional letter, followed by a dot and space.
    # Adjusted to handle rules like "Glossary" or "Credits" potentially
    rule_pattern = re.compile(r"^(?:(\d{1,3}(?:\.\d{1,2}[a-z]?)?)\.\s+)|(Glossary|Credits)")

    print(f"Parsing rules from: {file_path}")
    try:
        with open(file_path, "r", encoding='utf-8') as f: # Read as UTF-8
            for line in f:
                line = line.strip()
                if not line: # Skip empty lines
                    continue

                match = rule_pattern.match(line)
                if match:
                    # If we were processing a previous rule, save it
                    if current_rule_id and current_rule_text:
                        rules_data.append({
                            "id": current_rule_id,
                            "text": current_rule_text.strip()
                        })
                        # print(f"Parsed rule: {current_rule_id}") # Uncomment for verbose parsing output

                    # Start the new rule
                    # Group 1 captures numbered rules, Group 2 captures Glossary/Credits
                    current_rule_id = match.group(1) or match.group(2)
                    # Remove the rule number/title from the beginning of the text
                    current_rule_text = rule_pattern.sub('', line, count=1).strip()

                elif current_rule_id:
                    # If it's not a new rule start, append the line to the current rule text
                    current_rule_text += "\n" + line

            # Add the last rule after the loop finishes
            if current_rule_id and current_rule_text:
                rules_data.append({
                    "id": current_rule_id,
                    "text": current_rule_text.strip()
                })
                # print(f"Parsed rule: {current_rule_id}") # Uncomment for verbose parsing output

        print(f"Successfully parsed {len(rules_data)} rules sections.")
        return rules_data

    except FileNotFoundError:
        print(f"Error: Rules file not found at {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return []


def vectorize_and_store(rules_data: list[dict[str, str]], db_path: str, collection_name: str):
    """
    Vectorizes the parsed rules and stores them in a ChromaDB collection.

    Args:
        rules_data: A list of rule dictionaries ({'id': str, 'text': str}).
        db_path: The directory path to store the persistent ChromaDB database.
        collection_name: The name for the ChromaDB collection.
    """
    if not rules_data:
        print("No rules data provided to vectorize.")
        return

    print(f"Initializing ChromaDB client at: {db_path}")
    # Ensure the directory exists
    os.makedirs(db_path, exist_ok=True)

    # Use the default embedding function (requires sentence-transformers)
    # You can explicitly set one if needed:
    # ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    # client = chromadb.PersistentClient(path=db_path, embedding_function=ef) # If specifying EF
    ef = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    client = chromadb.PersistentClient(path=db_path, embedding_functions=ef)

    print(f"Getting or creating ChromaDB collection: {collection_name}")
    # Use get_or_create_collection to avoid errors if the collection already exists
    # If you need to explicitly set the embedding function, pass it here:
    # collection = client.get_or_create_collection(name=collection_name, embedding_function=ef)
    collection = client.get_or_create_collection(name=collection_name)

    print(f"Preparing {len(rules_data)} rules for vectorization and storage...")
    ids = [rule["id"] for rule in rules_data]
    documents = [rule["text"] for rule in rules_data]

    # Add data to ChromaDB in batches to avoid potential memory issues with very large datasets
    batch_size = 100 # Adjust batch size as needed
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_documents = documents[i:i + batch_size]

        print(f"Adding batch {i // batch_size + 1} ({len(batch_ids)} rules) to ChromaDB...")
        try:
            collection.add(
                documents=batch_documents,
                ids=batch_ids
            )
        except Exception as e:
            print(f"Error adding batch {i // batch_size + 1} to ChromaDB: {e}")
            # Consider adding more robust error handling here (e.g., retries, logging failed IDs)

    print(f"Successfully added/updated {len(ids)} rules in the '{collection_name}' collection.")
    print(f"Database stored at: {db_path}")

# --- Main Execution ---

if __name__ == "__main__":
    # 2. Parse the rules file
    parsed_rules = parse_rules(LOCAL_RULES_FILE)

    # 3. Vectorize and store the rules in ChromaDB
    if parsed_rules:
        vectorize_and_store(parsed_rules, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)
    else:
        print("Rule parsing failed. Skipping vectorization.")

    print("\nScript finished.")
    # Example of how to query the database (optional)
    # client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    # collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    # results = collection.query(
    #     query_texts=["What is banding?"],
    #     n_results=5
    # )
    # print("\nExample Query Results for 'What is banding?':")
    # print(results)

