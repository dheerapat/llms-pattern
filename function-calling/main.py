import os
import json
import requests
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionToolParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from dotenv import load_dotenv
from transformers.utils.chat_template_utils import get_json_schema

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)


def search_gutenberg_books(search_terms: list[str]):
    """
    Search for books in the Project Gutenberg library based on specified search terms

    Args:
        search_terms: A list of search keywords
    """
    search_query = " ".join(search_terms)
    url = "https://gutendex.com/books"
    response = requests.get(url, params={"search": search_query})
    simplified_results = []
    for book in response.json().get("results", []):
        simplified_results.append(
            {
                "id": book.get("id"),
                "title": book.get("title"),
                "authors": book.get("authors"),
            }
        )
    return simplified_results


schema = get_json_schema(search_gutenberg_books)
tool_param = ChatCompletionToolParam(**schema)
tools = [tool_param]
TOOL_MAPPING = {"search_gutenberg_books": search_gutenberg_books}


def main(instruction: str):
    conversation = [
        ChatCompletionSystemMessageParam(
            role="system",
            content="You are helpful assistance that can access index of Gutenberg project to find an information about public domain books.",
        ),
        ChatCompletionUserMessageParam(role="user", content=instruction),
    ]

    completion = client.chat.completions.create(
        model=os.getenv("TEXT_MODEL_NAME", ""),
        messages=conversation,
        tool_choice="required",
        tools=tools,
    )

    response = completion.choices[0].message

    if response.tool_calls is None:
        return

    conversation.append(response)

    for tool_call in response.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool_response = TOOL_MAPPING[tool_name](**tool_args)
        conversation.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(tool_response),
            }
        )
    answer = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", ""), messages=conversation
    )
    print(answer.choices[0].message.content)


if __name__ == "__main__":
    main("What are the title of book from Homer?")
