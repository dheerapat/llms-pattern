import requests
import xml.etree.ElementTree as ET
from pydantic import BaseModel, Field
from typing import List, Optional


class Header(BaseModel):
    type: str
    version: str


class ESearchResult(BaseModel):
    idlist: List[str]


class PubMedSearchResponse(BaseModel):
    header: Header
    esearchresult: ESearchResult


class AbstractText(BaseModel):
    label: Optional[str] = None
    content: str = Field(alias="text")

    class Config:
        populate_by_name = True


class Abstract(BaseModel):
    title: str
    abstract_sections: List[AbstractText] = Field(default_factory=list)
    doi: Optional[str] = None

    @property
    def full_abstract(self) -> str:
        """Combine all abstract sections into a single text."""
        if not self.abstract_sections:
            return ""

        # Join sections with labels if available
        parts = []
        for section in self.abstract_sections:
            if section.label:
                parts.append(f"{section.label}: {section.content}")
            else:
                parts.append(section.content)

        return " ".join(parts)


class PubMedArticle(BaseModel):
    pmid: str
    abstact: Abstract


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
    print(f"Search Results:")
    print(f"Article ids: {result.esearchresult.idlist}")


def get_abstract(id: str, mode: str):
    param = {"db": "pubmed", "id": id, "retmode": mode, "rettype": "abstract"}

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


def parse_pubmed_xml(xml_content: str) -> List[PubMedArticle]:
    try:
        root = ET.fromstring(xml_content)
        articles = []

        for pubmed_article in root.findall(".//PubmedArticle"):
            pmid_element = pubmed_article.find(".//PMID")
            pmid = pmid_element.text if pmid_element is not None else ""

            title_element = pubmed_article.find(".//ArticleTitle")
            title = title_element.text if title_element is not None else ""

            abstract_sections = []
            abstract_element = pubmed_article.find(".//Abstract")

            if abstract_element is not None:
                for abstract_text in abstract_element.findall(".//AbstractText"):
                    label = abstract_text.get("Label")
                    content = abstract_text.text if abstract_text.text else ""

                    abstract_sections.append(AbstractText(label=label, text=content))

            doi = None
            article_id_list = pubmed_article.find(".//ArticleIdList")

            if article_id_list is not None:
                for article_id in article_id_list.findall(".//ArticleId"):
                    if article_id.get("IdType") == "doi":
                        doi = article_id.text
                        break

            article = Abstract(
                title=title or "", abstract_sections=abstract_sections, doi=doi
            )

            articles.append(PubMedArticle(pmid=pmid or "", abstact=article))

        return articles

    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML: {e}")


def print_article_info(articles: List[PubMedArticle]) -> None:
    for article in articles:
        print(f"\n{'='*80}")
        print(f"PMID: {article.pmid}")
        print(f"Title: {article.abstact.title}")
        print(f"DOI: {article.abstact.doi}")
        print(f"\nAbstract:")

        if article.abstact.abstract_sections:
            for section in article.abstact.abstract_sections:
                if section.label:
                    print(f"\n{section.label.upper()}:")
                    print(section.content)
                else:
                    print(section.content)
        else:
            print("No abstract available")

        print(f"\nFull Abstract (combined):")
        print(article.abstact.full_abstract)


if __name__ == "__main__":
    try:
        result = search_journal("bilastine allergic rhinitis")
        print_search_results(result)

        xml = get_abstract("99999999999", "xml")
        articles = parse_pubmed_xml(xml)
        print_article_info(articles)
    except Exception as e:
        print(f"Error: {e}")
