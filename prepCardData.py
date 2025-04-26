import json
import re

def main():
    file_path = 'data/cardData.json'
    print(file_path)
    create_json_elements(file_path,"data/cardsFiltered.json")
    print("Done")

def create_metadata(element):
    outputElement = {
        "name": element["name"],
        "type": element["type_line"],
        "set": element["set_name"]
    }
    # don't forget legalities
    
    if "loyalty" in element:
        outputElement["loyalty"] = element["loyalty"]
    if "power" in element:
        outputElement["power"] = element["power"]
    if "toughness" in element:
        outputElement["toughness"] = element["toughness"]
    if "oracle_text" in element:
        outputElement["oracle_text"] = element["oracle_text"]
    if "mana_cost" in element:
        outputElement["mana_cost"] = element["mana_cost"]
    if "cmc" in element:
        outputElement["cmc"] = element["cmc"]
    if "legalities" in element:
        outputElement["legalities"] = element["legalities"]
    return outputElement

def create_json_elements(file_path, output_file):
    with open(file_path, 'r',  encoding="utf-8" ) as file:
        data = json.load(file)
        with open(output_file, 'w') as outfile:
            root = []
            for element in data:
                metadata = create_metadata(element)
                outputElement = {
                    "metadata" : metadata,
                    "document" : prepare_card_text_for_embedding(metadata)
                }
                
                root.append(outputElement)
            json.dump(root,outfile, indent=4, separators=(',', ': ')) 

def prepare_card_text_for_embedding(card_data, include_flavor_text=False):
    """
    Prepares a structured text document for a single Magic: The Gathering card,
    suitable for generating embeddings.

    Args:
        card_data (dict): A dictionary containing card information. Expected keys:
                          'name', 'mana_cost', 'type_line', 'oracle_text',
                          'power', 'toughness', 'loyalty', 'flavor_text' (optional).
                          Missing keys will be handled gracefully.
        include_flavor_text (bool): Whether to include flavor text in the output.

    Returns:
        str: A formatted string representing the card's textual data.
    """
    if not isinstance(card_data, dict):
        raise TypeError("Input 'card_data' must be a dictionary.")

    # 1. Extract fields (handle missing keys safely)
    name = card_data.get('name', '').lower()
    mana_cost_symbols = card_data.get('mana_cost', '')
    type_line = card_data.get('type', '').lower()
    # Oracle text preprocessing happens later
    oracle_text = card_data.get('oracle_text', '')
    power = card_data.get('power')
    toughness = card_data.get('toughness')
    loyalty = card_data.get('loyalty')
    flavor_text = card_data.get('flavor_text', '') if include_flavor_text else ''

    # 2. Preprocess Mana Cost
    processed_mana_cost = preprocess_mana_cost(mana_cost_symbols)

    # 3. Preprocess Rules Text
    processed_oracle_text = preprocess_oracle_text(oracle_text)

    # 4. Format Stats
    stats_line = format_stats(power, toughness, loyalty)

    # 5. Assemble Document Parts
    document_parts = []
    if name:
        document_parts.append(f"Card Name: {name}")
    if processed_mana_cost != "none": # Only add if there's a cost
        document_parts.append(f"Mana Cost: {processed_mana_cost}")
    if type_line:
        document_parts.append(f"Type: {type_line}")
    if processed_oracle_text:
        # Add the "Rules Text:" label only if there is text
        document_parts.append(f"Rules Text:\n{processed_oracle_text}")
    if stats_line != "none": # Only add if there are stats
        document_parts.append(f"Stats: {stats_line}")

    # 6. Add optional Flavor Text
    if include_flavor_text and flavor_text:
        # Optional: further preprocess flavor text (e.g., lowercase)
        processed_flavor_text = flavor_text.lower().strip()
        if processed_flavor_text:
            document_parts.append(f"Flavor Text: {processed_flavor_text}")

    # 7. Join parts with newlines
    # Filter ensures that if a section resulted in an empty string, it doesn't create extra newlines
    return "\n".join(filter(None, document_parts)).strip()

def format_stats(power, toughness, loyalty):
    """Formats Power/Toughness or Loyalty into a string."""
    if power is not None and toughness is not None:
        return f"power {power} toughness {toughness}"
    elif loyalty is not None:
        return f"loyalty {loyalty}"
    else:
        return "none"
    
def preprocess_oracle_text(oracle_text):
    """
    Preprocesses the rules text (oracle text).
    - Converts to lowercase.
    - Replaces common symbols ({T}, {Q}, {E}, etc.) with text.
    - Normalizes whitespace and newlines.
    - Optional: Could add reminder text removal here if desired.
    """
    if not oracle_text:
        return ""

    # Lowercase the text
    processed_text = oracle_text.lower()

    # Replace symbols within the text (similar to mana cost, but only for relevant symbols)
    # Use a subset of the symbol map or define specifically for rules text


    # Iteratively replace symbols found in the text
    # This simple replace might incorrectly replace parts of words if symbols are substrings.
    # A regex approach is safer for replacing only exact symbol matches.
    for symbol, text_repr in SYMBOL_MAP.items():
         # Use regex to replace only the exact symbol, handling potential regex special chars in symbol
         processed_text = re.sub(re.escape(symbol), text_repr, processed_text)


    # --- Optional: Reminder Text Removal ---
    # This regex removes text within parentheses. Be cautious as some rules use parentheses.
    # Consider if your data source provides rules text without reminders separately.
    # processed_text = re.sub(r'\([^)]*\)', '', processed_text).strip()
    # ---------------------------------------

    # Normalize whitespace: replace multiple spaces/newlines with single space,
    # then ensure abilities separated by original newlines are preserved.
    # Split by original newlines, strip whitespace from each line, join back with single newline.
    lines = [line.strip() for line in processed_text.split('\n')]
    processed_text = '\n'.join(filter(None, lines)) # Filter out empty lines

    return processed_text

def preprocess_mana_cost(mana_cost_symbols):
    """
    Translates mana symbols into a textual representation.
    Handles common mana symbols, generic mana, X, hybrid, Phyrexian, Tap, Untap, Snow, Energy.
    """
    if not mana_cost_symbols:
        return "none"
    # Find all symbols like {.} or {..} or {./.}
    symbols = re.findall(r'(\{.*?\})', mana_cost_symbols)
    processed_parts = []
    for symbol in symbols:
        processed_parts.append(SYMBOL_MAP.get(symbol, symbol)) # Keep unknown symbols as is
    return " ".join(processed_parts) if processed_parts else "none"

# Define mappings for symbols
# Order matters for multi-character symbols (e.g., {10} before {1})
SYMBOL_MAP = {
    # Basic Mana
    "{W}": "white mana",
    "{U}": "blue mana",
    "{B}": "black mana",
    "{R}": "red mana",
    "{G}": "green mana",
    "{C}": "colorless mana",
    # Generic Mana (handle specific numbers first)
    "{20}": "twenty generic mana",
    "{19}": "nineteen generic mana",
    "{18}": "eighteen generic mana",
    "{17}": "seventeen generic mana",
    "{16}": "sixteen generic mana",
    "{15}": "fifteen generic mana",
    "{14}": "fourteen generic mana",
    "{13}": "thirteen generic mana",
    "{12}": "twelve generic mana",
    "{11}": "eleven generic mana",
    "{10}": "ten generic mana",
    "{9}": "nine generic mana",
    "{8}": "eight generic mana",
    "{7}": "seven generic mana",
    "{6}": "six generic mana",
    "{5}": "five generic mana",
    "{4}": "four generic mana",
    "{3}": "three generic mana",
    "{2}": "two generic mana",
    "{1}": "one generic mana",
    "{0}": "zero generic mana",
    "{X}": "x generic mana",
    # Hybrid Mana
    "{W/U}": "white or blue mana",
    "{W/B}": "white or black mana",
    "{U/B}": "blue or black mana",
    "{U/R}": "blue or red mana",
    "{B/R}": "black or red mana",
    "{B/G}": "black or green mana",
    "{R/G}": "red or green mana",
    "{R/W}": "red or white mana",
    "{G/W}": "green or white mana",
    "{G/U}": "green or blue mana",
    # Phyrexian Mana
    "{W/P}": "phyrexian white mana",
    "{U/P}": "phyrexian blue mana",
    "{B/P}": "phyrexian black mana",
    "{R/P}": "phyrexian red mana",
    "{G/P}": "phyrexian green mana",
    "{C/P}": "phyrexian colorless mana", # Hypothetical, but good to include
    # Hybrid Phyrexian / Generic
    "{2/W}": "two generic or white mana",
    "{2/U}": "two generic or blue mana",
    "{2/B}": "two generic or black mana",
    "{2/R}": "two generic or red mana",
    "{2/G}": "two generic or green mana",
    # Other Symbols often found in costs or rules text
    "{T}": "tap symbol",
    "{Q}": "untap symbol",
    "{S}": "snow mana",
    "{E}": "energy counter symbol",
    # Add any other symbols you encounter ({CHAOS}, etc.)
}

if __name__ == "__main__":
    main()