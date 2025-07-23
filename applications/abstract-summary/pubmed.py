from pydantic import BaseModel
from typing import List
import requests


class Header(BaseModel):
    type: str
    version: str


class ESearchResult(BaseModel):
    idlist: List[str]


class PubMedSearchResponse(BaseModel):
    header: Header
    esearchresult: ESearchResult


def search_journal(keyword: str) -> PubMedSearchResponse:
    if not keyword.strip():
        raise ValueError("Keyword cannot be empty")

    param = {
        "db": "pubmed",
        "term": f'"{keyword}"[Title/Abstract:~5]',
        "retmode": "json",
        "sort": "pub_date",
        "retmax": 10,
    }

    try:
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=param,
            timeout=30,
        )
        response.raise_for_status()

        json_data = response.json()
        return PubMedSearchResponse(**json_data)

    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch data from PubMed API: {e}")


def print_search_results(result: PubMedSearchResponse) -> None:
    """Print search results in a formatted way."""
    print(f"Search Results:")
    print(f"Article ids: {result.esearchresult.idlist}")


def get_abstract(id: str):
    param = {"db": "pubmed", "id": id, "retmode": "text", "rettype": "abstract"}

    try:
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=param,
            timeout=30,
        )
        response.raise_for_status()

        data = response.content
        text = data.decode("utf-8")
        return text

    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch data from PubMed API: {e}")


if __name__ == "__main__":
    try:
        result = search_journal("bilastine allergic rhinitis")
        print_search_results(result)

        text = get_abstract("22185044")
        print(text)

    except Exception as e:
        print(f"Error: {e}")
