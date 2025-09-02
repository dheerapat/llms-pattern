import requests
from bs4 import BeautifulSoup

url = "https://www.ncbi.nlm.nih.gov/books/NBK441824/"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.google.com/",
}

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    html_content = response.text
    soup = BeautifulSoup(html_content, "html.parser")
    document_div = soup.find("div", class_="document")
    reference_div = soup.find("div", class_="bk_tt")

    if document_div:
        print(document_div.get_text().strip())
    else:
        print("Error: The div with class 'document' was not found.")

    if reference_div:
        print(reference_div.get_text().strip())
    else:
        print("Error: The div with class 'bk_tt' was not found.")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
