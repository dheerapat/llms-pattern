import os
from typing import Literal, List
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from pubmed import get_abstract, search_journal

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)


class SearchKeyword(BaseModel):
    keyword: str


class RCTClassification(BaseModel):
    classification: Literal["rct", "not_rct_or_meta", "meta_analysis"]


def generate_keyword(query: str):
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
        temperature=0.1,
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
                Your task is to classify a given abstract into one of three categories based on the study design:
                
                ## Classification Categories:
                
                **1. RCT (Randomized Controlled Trial)**
                An RCT must have ALL of the following criteria:
                - Random assignment of participants to intervention groups
                - Human subjects (not animals, cells, or in vitro studies)
                - A control or comparison group
                - An intervention being tested
                
                **2. META_ANALYSIS**
                A meta-analysis must have ALL of the following criteria:
                - Systematic review of multiple studies
                - Quantitative synthesis/statistical pooling of results from multiple studies
                - Analysis combining data from at least 2 studies
                - Clear methodology for study selection and data extraction
                
                **3. NOT_RCT_OR_META**
                Any study that does not meet the criteria for either RCT or meta-analysis, including:
                - Observational studies (cohort, case-control, cross-sectional)
                - Case reports or case series
                - Animal studies or in vitro research
                - Single-arm trials without randomization
                - Systematic reviews without quantitative synthesis
                - Editorial, commentary, or review articles
                
                ## Classification Rules:
                - Classify as "RCT" only if most RCT criteria are met
                - Classify as "META_ANALYSIS" only if most meta-analysis criteria are met
                - If a study equally meets criteria for both (rare), prioritize "META_ANALYSIS"
                - Otherwise, classify as "NOT_RCT_OR_META"
                Provide your classification as one of: "rct", "meta_analysis", or "not_rct_or_meta"
                """,
            },
            {"role": "user", "content": abstract_text},
        ],
        response_format=RCTClassification,
        temperature=0.1,
    )
    result = response.choices[0].message.parsed
    if result is None:
        raise ValueError("Failed to parse response from OpenAI API")
    return result


if __name__ == "__main__":
    output = generate_keyword("benefit of circumcision")
    print(output)

    rct_abstract = get_abstract("40419146", "text")  # rct
    print(classify_rct(rct_abstract))

    meta_abstract = get_abstract("36380619", "text")  # meta-analysis
    print(classify_rct(meta_abstract))

    not_rct_abstract = get_abstract("22686617", "text")  # not rct
    print(classify_rct(not_rct_abstract))
