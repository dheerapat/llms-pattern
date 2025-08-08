import os
import typer
import json
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
from rich import print

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)


class MultipleChoiceQuestionFormat(BaseModel):
    reasoning: str = Field(
        description="your step by step reasoning to construct the question"
    )
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
                "content": "You are tutor expert, you can generating multiple choice question to help student learn about various topic",
            },
            {"role": "user", "content": instruction},
        ],
        response_format=MultipleChoiceQuestionFormat,
        temperature=0.2,
    )
    print(response)
    return response.choices[0].message.content


def response(instruction: str):

    response = client.responses.parse(
        model=os.getenv("TEXT_MODEL_NAME", ""),
        input=[
            {
                "role": "system",
                "content": "You are tutor expert, you can generating multiple choice question to help student learn about various topic",
            },
            {"role": "user", "content": instruction},
        ],
        text_format=MultipleChoiceQuestionFormat,
        temperature=0.2,
    )

    return response.output_parsed


def main():
    result = make_request(
        "create a multiple choice question to help student learn about cellular respiration"
    )
    if result is not None:
        print(result)


if __name__ == "__main__":
    typer.run(main)
