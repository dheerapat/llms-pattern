import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)


def main():
    completion = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", ""),
        messages=[
            {"role": "user", "content": "I'm pickle Rick!"},
        ],
    )
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
