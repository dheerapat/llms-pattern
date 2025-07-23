import os
from typing import Literal
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
    classification: Literal["rct", "not_rct"]


def generate_keyword(query: str):
    response = client.beta.chat.completions.parse(
        model=os.getenv("TEXT_MODEL_NAME", ""),
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert in search optimization and information retrieval.
                Your primary function is to extract and generate optimal search keywords from user queries for use in search engines and vector databases.
                """,
            },
            {"role": "user", "content": query},
        ],
        response_format=SearchKeyword,
        temperature=0.2,
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
                You are an expert medical researcher specializing in clinical trial methodology.
                Your task is to determine if a given abstract describes a randomized controlled trial (RCT) conducted in human subjects.
                
                An RCT must have:
                1. Random assignment of participants to intervention groups
                2. Human subjects (not animals, cells, or in vitro studies)
                3. A control or comparison group
                4. An intervention being tested
                
                Classify as "rct" only if all criteria are met. Otherwise, classify as "not_rct".
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
    output = generate_keyword("is ketogenic diet help in weight loss")
    print(output)

    rct_abstract = get_abstract("39376275")  # rct
    print(classify_rct(rct_abstract))

    not_rct_abstract = get_abstract("22686617")  # not rct
    print(classify_rct(not_rct_abstract))
