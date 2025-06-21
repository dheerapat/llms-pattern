import os
import typer
from rich import print
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)


def chat_loop():
    print("type 'exit' to exit the app")
    chat = []
    while True:
        # print(chat)
        user_input = typer.prompt("user")
        if user_input.lower() == "exit":
            print("Bye!")
            break
        chat.append({"role": "user", "content": user_input})
        completion = client.chat.completions.create(
            model=os.getenv("TEXT_MODEL_NAME", ""), messages=chat, temperature=0.2
        )
        print(f"assistant: {completion.choices[0].message.content}")
        chat.append(
            {"role": "assistant", "content": completion.choices[0].message.content}
        )


if __name__ == "__main__":
    typer.run(chat_loop)
