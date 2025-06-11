import os
import typer
from rich import print
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


def main():
    print("type 'exit' to exit the app")
    chat = []
    while True:
        user_input = typer.prompt("user")
        if user_input.lower() == "exit":
            break
        chat.append({"role": "user", "content": user_input})
        completion = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14", messages=chat, temperature=0.2
        )
        print(f"assistant: {completion.choices[0].message.content}")
        chat.append(
            {"role": "assistant", "content": completion.choices[0].message.content}
        )
    # print(chat)


if __name__ == "__main__":
    typer.run(main)
