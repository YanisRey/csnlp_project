import requests
from bs4 import BeautifulSoup
import json  # Import the json module for saving the data

# Base URL for the Wikipedia page
base_url = "https://en.wikipedia.org"

# URL of the main page containing links to subpages
main_page_url = base_url + "/wiki/Wikipedia:Lists_of_common_misspellings"

# Fetch the main page
response = requests.get(main_page_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Initialize a dictionary to store correct words and their misspellings
misspellings_dict = {}

# Find all links to the subpages (A, B, C, etc.)
subpage_links = soup.select('a[href^="/wiki/Wikipedia:Lists_of_common_misspellings/"]')

# Iterate through each subpage link
for link in subpage_links:
    subpage_url = base_url + link['href']
    subpage_response = requests.get(subpage_url)
    subpage_soup = BeautifulSoup(subpage_response.text, 'html.parser')

    # Find the content area of the subpage
    content = subpage_soup.find('div', {'class': 'mw-parser-output'})
    if content:
        # Extract all list items (misspellings)
        for li in content.find_all('li'):
            text = li.get_text(strip=True)
            if '(' in text and ')' in text:
                # Split into misspelling and correct spelling
                misspelling, correct = text.split('(', 1)
                misspelling = misspelling.strip()
                correct = correct.split(')', 1)[0].strip()

                # Add to the dictionary
                if correct in misspellings_dict:
                    misspellings_dict[correct].append(misspelling)
                else:
                    misspellings_dict[correct] = [misspelling]

# Save the dictionary to a JSON file
with open('../../data/misspellings/misspellings.json', 'w') as f:
    json.dump(misspellings_dict, f, indent=4)

print("Misspellings saved to '../../data/misspellings/misspellings.json'")