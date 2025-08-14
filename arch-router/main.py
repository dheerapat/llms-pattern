import os
import json
from openai import OpenAI
from typing import Any, Dict, List
from dotenv import load_dotenv
from rich import print

load_dotenv()

# use arch-router-1.5b.gguf model from katanemo
client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

TASK_INSTRUCTION = """
You are a helpful assistant designed to find the best suited route.
You are provided with route description within <routes></routes> XML tags:
<routes>

{routes}

</routes>

<conversation>

{conversation}

</conversation>
"""

FORMAT_PROMPT = """
Your task is to decide which route is best suit with user intent on the conversation in <conversation></conversation> XML tags.  Follow the instruction:
1. If the latest intent from user is irrelevant or user intent is full filled, response with other route {"route": "other"}.
2. You must analyze the route descriptions and find the best match route for user latest intent. 
3. You only response the name of the route that best matches the user's request, use the exact name in the <routes></routes>.

Based on your analysis, provide your response in the following JSON formats if you decide to match any route:
{"route": "route_name"} 
"""

# Define route config
route_config = [
    {
        "name": "code_generation",
        "description": "Generating new code snippets, functions, or boilerplate based on user prompts or requirements",
    },
    {
        "name": "bug_fixing",
        "description": "Identifying and fixing errors or bugs in the provided code across different programming languages",
    },
    {
        "name": "performance_optimization",
        "description": "Suggesting improvements to make code more efficient, readable, or scalable",
    },
    {
        "name": "api_help",
        "description": "Assisting with understanding or integrating external APIs and libraries",
    },
    {
        "name": "programming",
        "description": "Answering general programming questions, theory, or best practices",
    },
]


def format_prompt(
    route_config: List[Dict[str, Any]], conversation: List[Dict[str, Any]]
):
    return (
        TASK_INSTRUCTION.format(
            routes=json.dumps(route_config), conversation=json.dumps(conversation)
        )
        + FORMAT_PROMPT
    )


conversation = [
    {
        "role": "user",
        "content": "explain clean architecture",
    }
]

route_prompt = format_prompt(route_config, conversation)
print(route_prompt)

completion = client.chat.completions.create(
    model=os.getenv("TEXT_MODEL_NAME", ""),
    messages=[
        {"role": "user", "content": route_prompt},
    ],
)

print(f"{completion.choices[0].message.content}")
