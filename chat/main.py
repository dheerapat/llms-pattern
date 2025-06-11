import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


def main():
    completion = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=[
            {"role": "user", "content": "I'm pickle Rick!"},
        ],
    )
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
