import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

current_directory = os.path.dirname(os.path.abspath(__file__))
client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main(img: str):
    img_path = os.path.join(current_directory, "img", img)
    base64_image = encode_image(img_path)
    completion = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", ""),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarize this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        temperature=0.2,
    )
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main("symlink.jpg")
