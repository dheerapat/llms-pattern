import os
from typing import List, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from pubmed import PubMedArticle, get_abstract, parse_pubmed_xml, search_journal

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)


class SearchKeyword(BaseModel):
    reasoning: str = Field(
        description="your step by step reasoning on how to get these keyword"
    )
    keyword: str


class RCTClassification(BaseModel):
    reasoning: str = Field(description="your step by step reasoning on your decision")
    classification: Literal["rct", "animal_studies", "meta_analysis", "review_article"]


def generate_keyword(query: str) -> SearchKeyword:
    response = client.beta.chat.completions.parse(
        model=os.getenv("TEXT_MODEL_NAME", ""),
        messages=[
            {
                "role": "system",
                "content": """
                You are a biomedical search specialist focused on generating optimal keywords for PubMed database searches.

                ## Primary Function
                Convert user queries into a single, concise search keyword that maximizes relevant PubMed results when used with the `[Title/Abstract:~5]` proximity operator.
                
                ## Keyword Requirements
                - **Single string output**: Return exactly one keyword or phrase
                - **2-4 words maximum**: Keep it concise for effective proximity searching
                - **Medical terminology**: Use established biomedical terms when possible
                - **PubMed optimized**: Consider how terms appear in medical literature titles/abstracts
                
                ## Optimization Strategy
                
                ### Query Analysis
                1. **Identify core medical concept** from user's question
                2. **Use precise terminology** (e.g., "myocardial infarction" vs "heart attack")
                3. **Consider search scope** - specific enough to be relevant, broad enough to return results
                4. **Account for proximity search** - terms should logically appear near each other in abstracts
                
                ### Keyword Selection Rules
                - **Primary concept first**: Most important medical term
                - **2-3 word phrases**: Work well with proximity operator (~5)
                - **Avoid stop words**: Exclude "and", "or", "the", "of" when possible
                - **Use established terms**: Prefer MeSH terminology and standard medical language
                - **Consider synonyms**: Choose the most commonly used variant in literature
                
                ### Common Patterns
                - **Disease + Treatment**: "diabetes insulin"
                - **Procedure + Outcome**: "surgery complications" 
                - **Drug + Effect**: "aspirin cardioprotection"
                - **Condition + Risk**: "obesity cardiovascular"
                
                ## Examples
                | User Query | Optimal Keyword |
                |------------|-----------------|
                | "Side effects of statins on muscles" | "statin myopathy" |
                | "How effective is cognitive behavioral therapy for depression?" | "CBT depression" |
                | "Cancer treatment using immunotherapy" | "cancer immunotherapy" |
                
                ## Output Format
                Return only the keyword string - no explanation, quotes, or additional text.
                
                ## Quality Checks
                - Would this keyword appear in relevant PubMed titles/abstracts?
                - Is it specific enough to avoid irrelevant results?
                - Does it use terminology medical researchers would use?
                - Will the proximity operator (~5) work effectively with these terms?
                """,
            },
            {"role": "user", "content": query},
        ],
        response_format=SearchKeyword,
        temperature=1,
    )
    result = response.choices[0].message.parsed
    if result is None:
        raise ValueError("Failed to parse response from OpenAI API")
    return result


def classify_rct(abstract_text: str) -> RCTClassification:
    response = client.beta.chat.completions.parse(
        model=os.getenv("TEXT_MODEL_NAME", ""),
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert medical researcher specializing in clinical trial methodology and systematic review analysis.
                Your task is to classify a given abstract into one of four categories based on the study design and methodology.
                
                ## Classification Categories:
                
                **1. RCT (Randomized Controlled Trial)**
                A study must have ALL of the following criteria to be classified as RCT:
                - Use a conservative approach with this category. If not sure, don't answer this
                - Random assignment/randomization of participants to intervention groups
                - Human subjects (not animals, cells, or in vitro studies)
                - A control or comparison group (placebo, standard care, or active comparator)
                - An intervention being tested (drug, procedure, therapy, etc.)
                - Prospective design with outcome measurement
                
                **2. META_ANALYSIS**
                A study must have ALL of the following criteria to be classified as meta-analysis:
                - Use a conservative approach with this category. If not sure, don't answer this
                - Systematic review methodology with clearly defined search strategy
                - Quantitative synthesis/statistical pooling of results from multiple studies
                - Analysis combining numerical data from at least 2 independent RCT studies
                - Forest plots, pooled effect sizes, or combined statistical measures
                - Assessment of study quality or risk of bias
                
                **3. ANIMAL_STUDIES**
                A study must have ALL of the following criteria to be classified as animal study:
                - Primary subjects are animals (rodents, primates, fish, birds, etc.)
                - Experimental design with random assignment to control and treatment groups
                - Quantitative outcomes with statistical analysis of results
                - Investigation of biological mechanisms, drug effects, or interventions
                
                **4. REVIEW_ARTICLE**
                Any study that does not meet the strict criteria for the above three categories, including:
                - Narrative reviews or expert opinions without systematic methodology
                - Observational studies (cohort, case-control, cross-sectional)
                - Case reports or case series
                - Editorial or commentary articles
                - Studies that mention other research but don't conduct systematic review or meta-analysis
                - In vitro or cell culture studies
                - Surveys or questionnaire-based studies without experimental intervention
                
                ## Important Classification Notes:
                - Be strict about RCT criteria - studies must explicitly mention randomization
                - Meta-analyses must show quantitative pooling, not just narrative synthesis
                - Animal studies may use randomization but are NOT RCTs - don't conflate the two.
                - Animal studies must be experimental, not just observational
                - When in doubt between categories, default to the most conservative classification
                - Look for key methodological terms: "randomized", "meta-analysis", "pooled", "animal model"
                
                ## Decision Process:
                1. First check if it's a meta-analysis (systematic + quantitative pooling)
                2. Then check if it's an RCT (human + randomized + controlled)
                3. Then check if it's an animal study (animal subjects + experimental)
                4. If none of the above, classify as review article
                """,
            },
            {"role": "user", "content": abstract_text},
        ],
        response_format=RCTClassification,
        temperature=1,
    )
    result = response.choices[0].message.parsed
    if result is None:
        raise ValueError("Failed to parse response from OpenAI API")
    return result


def generate_answer(abstracts: List[PubMedArticle], query: str) -> str:
    # Format abstracts into a readable string
    formatted_abstracts = []
    for i, article in enumerate(abstracts, 1):
        abstract_text = f"""
        Article {i} (PMID: {article.pmid}):
        Title: {article.abstract.title}
        Abstract: {article.abstract.full_abstract}
        DOI: {article.abstract.doi or 'Not available'}""".strip()
        formatted_abstracts.append(abstract_text)

    abstracts_content = "\n\n" + "\n\n".join(formatted_abstracts)

    completion = client.chat.completions.create(
        model=os.getenv("TEXT_MODEL_NAME", ""),
        messages=[
            {
                "role": "system",
                "content": f"""
                You are an expert medical assistant specializing in clinical decision support using evidence-based research.
                Your task is to analyze the provided abstracts that come from searching the PubMed Entrez database using the user's query
                and answer the user's question based on the provided abstracts given to you.
                
                Please provide accurate, evidence-based responses and cite the relevant articles by their PMID when making specific claims.
                If the abstracts don't contain sufficient information to answer the query, please state this clearly.
                
                Abstracts:
                {abstracts_content}

                Your answer will conclude this conversation with the user. Please do not ask follow-up questions or offer additional assistance.
                """.strip(),
            },
            {"role": "user", "content": query},
        ],
    )
    result = completion.choices[0].message.content or ""
    return result


if __name__ == "__main__":
    # query = "allregic rhinitis firstline therapy"
    # keyword = generate_keyword(query)
    # print(keyword)

    # journal_ids = search_journal(keyword=keyword.keyword).esearchresult.idlist
    # ids = ", ".join(journal_ids)
    # print(ids)

    # abs = get_abstract(ids, "xml")
    # parsed = parse_pubmed_xml(abs)
    # for abstract in parsed:
    #     if abstract.abstract.full_abstract.strip():
    #         print("---")
    #         print(abstract.abstract.full_abstract)

    # result = generate_answer(parsed, query)
    # print("===")
    # print(result)

    # ids = ["36578889", "30039871", "35082662", "35537861", "33999947"]  # meta
    # ids = ["28066101","34476568","35939311","37960261","35956364"] # rct
    ids = ["32457512", "38720498", "21831011", "34706925", "37571305"]  # non rct

    for id in ids:
        print(id)
        abs = get_abstract(id, "xml")
        parsed = parse_pubmed_xml(abs)
        print(classify_rct(parsed[0].abstract.full_abstract))
