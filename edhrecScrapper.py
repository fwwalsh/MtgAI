import requests
from bs4 import BeautifulSoup
import time
import json
import random
import re
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions

# --- Configuration ---
BASE_URL = "https://edhrec.com"
COMMANDERS_URL = f"{BASE_URL}/commanders"
# Define a realistic User-Agent
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
# Number of commanders to scrape
COMMANDER_LIMIT = 100
# Delay between requests (in seconds) - BE RESPECTFUL! Increase if needed.
MIN_DELAY = 1
MAX_DELAY = 2
# Output file name
OUTPUT_FILE = "edhrec_top_commanders.json"

def main():
    # --- Main Script Logic ---

    all_commanders_data = []
    commander_urls = set() # Use a set to avoid duplicates if pagination logic overlaps
    # Start with the main commanders page
    current_url = COMMANDERS_URL

    # --- Loop through pages to find commander links ---
    # NOTE: EDHREC pagination might use JavaScript or different URL patterns.
    # This simple loop might need significant adjustment. Inspect how pagination works.
    page_count = 1
    max_pages = 20 # Limit pages to prevent infinite loops if logic fails
    print(f"Starting commander URL discovery...")


    while len(commander_urls) < COMMANDER_LIMIT and page_count <= max_pages:
        print(f"\nProcessing commander list page {page_count}: {current_url}")
        soup = get_soup(current_url)
        if not soup:
            print(f" Failed to get soup for page {page_count}. Stopping URL discovery.")
            break # Stop if a page fails

        # Find links to individual commander pages (Example selector - **NEEDS VERIFICATION**)
        # Look for 'a' tags within specific containers that hold commander names/links
        links_found_on_page = 0
        # Example: Find links within a grid/list container
        commander_list_container = soup.find('div', class_=re.compile(r'Grid_grid__EAPIs')) # Example
        temp = commander_list_container.find('div', class_="Card_container__Ng56K").find_all('div',class_="lazyload-wrapper")
        
        if commander_list_container:
            potential_links = commander_list_container.find_all('a', href=re.compile(r'^/commanders/'), recursive=True)
            for link in potential_links:
                href = link.get('href')
                if href:
                    full_url = urljoin(BASE_URL, href)
                    if full_url not in commander_urls:
                        commander_urls.add(full_url)
                        links_found_on_page += 1
                        if len(commander_urls) >= COMMANDER_LIMIT:
                            break # Stop adding once limit is reached
            print(f"  Found {links_found_on_page} new commander URLs on this page.")
        else:
            print("  Warning: Could not find commander list container using current selector.")

        if len(commander_urls) >= COMMANDER_LIMIT:
            print(f"\nReached commander limit ({COMMANDER_LIMIT}).")
            break

        # Find the 'next' page link (Example selector - **NEEDS VERIFICATION**)
        next_page_tag = soup.find('a', class_=re.compile(r'Pagination_next__'), href=True) # Example
        if next_page_tag:
            next_page_url = urljoin(BASE_URL, next_page_tag['href'])
            if next_page_url == current_url: # Avoid getting stuck on the same page
                print(" Next page URL is the same as current. Stopping pagination.")
                break
            current_url = next_page_url
            page_count += 1
        else:
            print(" Could not find 'next' page link. Stopping pagination.")
            break # Stop if no next page link is found

    print(f"\nDiscovered {len(commander_urls)} unique commander URLs.")

    # --- Scrape each commander's page ---
    print(f"\nStarting scraping individual commander pages (up to {COMMANDER_LIMIT})...")
    count = 0
    for url in list(commander_urls): # Iterate over a copy
        if count >= COMMANDER_LIMIT:
            break
        print(f"\nProcessing commander {count + 1}/{len(commander_urls)} (Target: {COMMANDER_LIMIT})")
        data = scrape_commander_page(url)
        if data:
            all_commanders_data.append(data)
            count += 1
        else:
            print(f"  Skipping commander due to previous error: {url}")

    print(f"\nSuccessfully scraped data for {len(all_commanders_data)} commanders.")

    # --- Save data to JSON ---
    print(f"\nSaving data to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_commanders_data, f, indent=4, ensure_ascii=False)
        print("Data saved successfully.")
    except IOError as e:
        print(f"Error saving data to file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")

    print("\nScript finished.")


# --- Helper Functions ---

def respectful_sleep():
    """Pauses execution for a random time between MIN_DELAY and MAX_DELAY."""
    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    print(f"    Sleeping for {delay:.2f} seconds...")
    time.sleep(delay)

chrome_options = ChromeOptions()
chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
chrome_options.add_argument("--no-sandbox")  # Bypass OS security model, needed for some environments
chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems in Docker
DRIVER = webdriver.Chrome(options=chrome_options)

def get_soup(url):

    """Fetches a URL and returns a BeautifulSoup object."""
    try:
        print(f"  Fetching URL: {url}")
        DRIVER.get(url) # Added headers to avoid bot detection
        
        lastHeight = DRIVER.execute_script("return document.body.scrollHeight")
        pause = 0.5
        while True:
            DRIVER.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause)
            newHeight = DRIVER.execute_script("return document.body.scrollHeight")
            if newHeight == lastHeight:
                break
            lastHeight = newHeight
        html = DRIVER.page_source
        
        #response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        soup = BeautifulSoup(html,  "html5lib")
        respectful_sleep() # Sleep *after* successful request
        return soup
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching {url}: {e}")
        respectful_sleep() # Sleep even on error to avoid hammering
        return None
    except Exception as e:
        print(f"  An unexpected error occurred while fetching {url}: {e}")
        respectful_sleep()
        return None

def parse_card_item(card_div):
    """Parses a card div to extract name, percentage, and synergy."""
    card_data = {}
    # Card Name (usually within an 'a' tag)
    name_tag = card_div.find('a', class_=re.compile(r'Card_cardLink__')) # Example selector, **NEEDS VERIFICATION**
    if name_tag:
        card_data['name'] = name_tag.get_text(strip=True)
    else: # Fallback if structure differs
        name_tag_fallback = card_div.find(class_=re.compile(r'Card_cardName__'))
        if name_tag_fallback:
             card_data['name'] = name_tag_fallback.get_text(strip=True)
        else:
             card_data['name'] = "Unknown Card Name" # Placeholder

    # Percentage and Synergy (often within specific divs/spans)
    # These selectors are HIGHLY LIKELY TO CHANGE - Inspect EDHREC's HTML
    percent_tag = card_div.find('div', class_=re.compile(r'Card_cardMetaPrimary__')) # Example
    synergy_tag = card_div.find('div', class_=re.compile(r'Card_cardMetaSecondary__')) # Example

    if percent_tag:
        # Extract percentage (might need regex or further parsing)
        match = re.search(r'(\d+)%', percent_tag.get_text())
        if match:
            card_data['percent_decks'] = int(match.group(1))
        else:
            card_data['percent_decks'] = None # Or some other indicator
    else:
        card_data['percent_decks'] = None

    if synergy_tag:
         # Extract synergy (might need regex or further parsing)
        match = re.search(r'([\+\-]?\d+)% Synergy', synergy_tag.get_text()) # Example text format
        if match:
            card_data['synergy'] = int(match.group(1))
        else:
            card_data['synergy'] = None
    else:
        card_data['synergy'] = None

    # Basic validation
    if card_data['name'] == "Unknown Card Name" and card_data['percent_decks'] is None and card_data['synergy'] is None:
        return None # Skip if no useful data found

    return card_data


def scrape_commander_page(commander_url):
    """Scrapes an individual commander's page for stats."""
    print(f" Scraping commander page: {commander_url}")
    soup = get_soup(commander_url)
    if not soup:
        return None

    commander_data = {}

    # --- Extract Commander Name (Example - verify selector) ---
    div = soup.find('div', class_=re.compile(r'CoolHeader_container__')) # Example selector
    name_tag =  div.find('h3')
    #name_tag = re.sub(r"\(Commander\)$","", div.find('h3').string)  
    commander_data['name'] = re.sub(r"\(Commander\)$","", name_tag.get_text(strip=True)) if name_tag else "Unknown Commander"
    commander_data['edhrec_url'] = commander_url

    # --- Extract Card Sections (Top Cards, High Synergy, etc.) ---
    # These selectors are CRITICAL and WILL LIKELY need adjustment.
    # Use your browser's developer tools (Inspect Element) on a commander's page.
    card_sections = soup.find_all('div', class_=re.compile(r'CardSections_cardSection__')) # Example container

    for section in card_sections:
        section_title_tag = section.find(['h2', 'h3'], class_=re.compile(r'CardSections_headerTitle__')) # Example
        section_title = section_title_tag.get_text(strip=True).lower() if section_title_tag else "unknown_section"

        # Normalize section titles (adjust based on actual titles on EDHREC)
        if "top cards" in section_title:
            key = "top_cards"
        elif "high synergy cards" in section_title:
            key = "high_synergy_cards"
        elif "new cards" in section_title:
            key = "new_cards"
        # Add more sections like 'top artifacts', 'top creatures' if needed
        else:
            key = section_title.replace(" ", "_") # Generic key

        cards_in_section = []
        # Find individual card containers within the section (Example selector)
        card_divs = section.find_all('div', class_=re.compile(r'Card_container__'))
        for card_div in card_divs:
            parsed_card = parse_card_item(card_div)
            if parsed_card:
                cards_in_section.append(parsed_card)

        if cards_in_section: # Only add section if cards were found
             commander_data[key] = cards_in_section

    # --- Extract Themes (Example - verify selector) ---
    themes_data = []
    themes_section = soup.find('div', id='themes') # Example ID selector
    if themes_section:
        theme_links = themes_section.find_all('a', href=re.compile(r'/themes/'))
        for link in theme_links:
             themes_data.append({
                 'name': link.get_text(strip=True),
                 'url': urljoin(BASE_URL, link.get('href'))
             })
    if themes_data:
        commander_data['themes'] = themes_data

    # --- Extract Tribes (Example - verify selector) ---
    tribes_data = []
    # Similar logic to themes, adjust selector based on actual HTML
    tribes_section = soup.find('div', id='tribes') # Example ID selector (might not exist)
    if tribes_section:
        tribe_links = tribes_section.find_all('a', href=re.compile(r'/tribes/'))
        for link in tribe_links:
            tribes_data.append({
                'name': link.get_text(strip=True),
                'url': urljoin(BASE_URL, link.get('href'))
            })
    if tribes_data:
        commander_data['tribes'] = tribes_data

    # --- Extract Salt Score (Example - verify selector) ---
    salt_score_tag = soup.find('div', class_=re.compile(r'SaltScore_saltScore__')) # Example
    if salt_score_tag:
        match = re.search(r'([\d\.]+)', salt_score_tag.get_text())
        if match:
            try:
                commander_data['salt_score'] = float(match.group(1))
            except ValueError:
                commander_data['salt_score'] = None
        else:
             commander_data['salt_score'] = None


    print(f"  Finished scraping {commander_data.get('name', 'Unknown')}. Found {len(commander_data.get('top_cards', []))} top cards, {len(commander_data.get('high_synergy_cards', []))} synergy cards.")
    return commander_data

if __name__ == "__main__":
    main()