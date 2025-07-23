import os
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)


class SearchKeyword(BaseModel):
    keyword: str


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


if __name__ == "__main__":
    output = generate_keyword("is ketogenic diet help in weight loss")
    print(output)
