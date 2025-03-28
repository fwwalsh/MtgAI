import json

def print_flat_elements(file_path, output_file):
    with open(file_path, 'r') as file:
        data = json.load(file)
        with open(output_file, 'w') as outfile:
            for element in data:
                outputElement = [
                    f"name: {element['name'].replace('/', ' ')}",
                    f"type: {element['type_line'].replace('/', ' ')}",

                    
                    f"set: {element['set_name']}"
                ]
                # don't forget legalities
                
                if "loyalty" in element:
                    outputElement.append(f"loyalty: {element['loyalty']}")
                if "power" in element:
                    outputElement.append(f"power: {element['power']}")
                if "toughness" in element:
                    outputElement.append(f"toughness: {element['toughness']}")
                if "oracle_text" in element:
                    outputElement.append(f"oracle_text: {element['oracle_text'].replace('\n', ' ')}")
                if "mana_cost" in element:
                    outputElement.append(f"mana_cost: {element['mana_cost']}")

                outfile.write(", ".join(outputElement) + '\n')

def print_json_elements(file_path, output_file):
    with open(file_path, 'r') as file:
        data = json.load(file)
        with open(output_file, 'w') as outfile:
            root = []
            for element in data:
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
                root.append(outputElement)
            json.dump(root,outfile, indent=4, separators=(',', ': ')) 

if __name__ == "__main__":
    file_path = 'oracle-cards-20250310210527.json'
    print(file_path)
    print_json_elements(file_path,"cardsFiltered.json")