import requests

# Configuration table for URLs and output filenames
DOWNLOAD_CONFIG = [
    {
        "url": "https://api.scryfall.com/bulk-data/oracle-cards",
        "output_filename": "cardData.json"
    },
    {
        "url": "https://media.wizards.com/2025/downloads/MagicCompRules%2020250404.txt",
        "output_filename": "cardRules.txt"
    }
]
DOWNLOAD_DIRECTORY = "./data"
def main():
    for config in DOWNLOAD_CONFIG:
        download_data(config["url"], f"{DOWNLOAD_DIRECTORY}/{config['output_filename']}")

def download_data(url, output_filename):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes

    if "scryfall" in url:
        bulk_data = response.json()
        # Find the 'default_cards' bulk data URL
        download_url = bulk_data['download_uri']# next( item['download_uri'] for item in bulk_data['data'])

        # Download the card data
        card_data_response = requests.get(download_url)
        card_data_response.raise_for_status()

        # Save to the specified output file
        with open(output_filename, "w", encoding="utf-8") as file:
            file.write(card_data_response.text)

        print(f"Card data downloaded and saved to {output_filename}")
    else:
        # Download the file directly for other URLs
        file_response = requests.get(url)
        file_response.raise_for_status()

        with open(output_filename, "w", encoding="utf-8") as file:
            file.write(file_response.text)

        print(f"File downloaded and saved to {output_filename}")


if __name__ == "__main__":
    main()