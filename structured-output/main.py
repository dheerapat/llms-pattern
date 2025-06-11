import os
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


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
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.beta.chat.completions.parse(
        model="gpt-4.1-nano-2025-04-14",
        messages=[
            {
                "role": "system",
                "content": "You are biology expert, you can generating multiple choice question to help student learn about biology",
            },
            {"role": "user", "content": instruction},
        ],
        response_format=MultipleChoiceQuestionFormat,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print(
        make_request(
            "create a sample question about biology to help student learn about dna replication"
        )
    )
