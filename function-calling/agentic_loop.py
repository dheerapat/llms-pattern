import os
import json
import requests
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionToolParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionMessageParam,
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


def call_llm(msgs: list[ChatCompletionMessageParam]):
    print(msgs)
    resp = client.chat.completions.create(
        model=os.getenv("TEXT_MODEL_NAME", ""), tools=tools, messages=msgs
    )
    return resp


def get_tool_response(response):
    tool_call = response.choices[0].message.tool_calls[0]
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)
    tool_result = TOOL_MAPPING[tool_name](**tool_args)

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_name,
        "content": json.dumps(tool_result),
    }


conversation = [
    ChatCompletionSystemMessageParam(
        role="system",
        content="You are helpful assistance that can access index of Gutenberg project to find an information about public domain books.",
    ),
    ChatCompletionUserMessageParam(
        role="user", content="What are the title of book from Homer?"
    ),
]

while True:
    resp = call_llm(conversation)
    conversation.append(resp.choices[0].message)
    if resp.choices[0].message.tool_calls is not None:
        conversation.append(get_tool_response(resp))
    else:
        break

print(conversation[-1].content)
