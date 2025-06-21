import os
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)


class MultipleChoiceQuestionFormat(BaseModel):
    question: str
    choice_a: str
    choice_b: str
    choice_c: str
    choice_d: str
    choice_e: str
    answer: Literal[
        "choice_a",
        "choice_b",
        "choice_c",
        "choice_d",
        "choice_e",
    ]


def make_request(instruction: str):

    response = client.beta.chat.completions.parse(
        model=os.getenv("TEXT_MODEL_NAME", ""),
        messages=[
            {
                "role": "system",
                "content": "You are biology expert, you can generating multiple choice question to help student learn about biology",
            },
            {"role": "user", "content": instruction},
        ],
        response_format=MultipleChoiceQuestionFormat,
        temperature=0.2,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print(
        make_request(
            "create a sample question about biology to help student learn about dna replication"
        )
    )
