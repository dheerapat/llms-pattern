import os
import base64
import typer
import pymupdf
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)


def pdf_to_image_converter(pdf_path: str, output_dir: str) -> int:
    """
    Convert PDF pages to images and save them in the specified output directory.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Directory to save the output images.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=120)  # type: ignore
        image_path = os.path.join(output_dir, f"page-{page.number}.png")
        pix.save(image_path)

    return len(doc)


def image_encoder(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def vllm_extractor(img: str, current_directory: str):
    img_path = os.path.join(current_directory, "img", img)
    base64_image = image_encoder(img_path)
    completion = client.chat.completions.create(
        model=os.getenv("VISION_MODEL_NAME", ""),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the content from the provided document image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        temperature=0.2,
    )
    print(completion.choices[0].message.content)
    response_content = completion.choices[0].message.content
    with open(os.path.join(current_directory, "resp.txt"), "a", encoding="utf-8") as f:
        f.write(f"{response_content}\n\n---\n\n")

    print("Response saved to resp.txt")


def llm_summarizer(file: str, current_directory: str):
    file_path = os.path.join(current_directory, file)
    with open(file_path, "rt") as f:
        completion = client.chat.completions.create(
            model=os.getenv("TEXT_MODEL_NAME", ""),
            messages=[
                {
                    "role": "system",
                    "content": "You are expert learner, you can summarized information down to a very important piece that really matters, and discard any not useful information.",
                },
                {
                    "role": "user",
                    "content": f"summarized following piece of information that extracted from document.\n{f.read()}",
                },
            ],
            temperature=0.2,
        )
        response_content = completion.choices[0].message.content
        return response_content


def main(pdf_path: str):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    img_directory = os.path.join(current_directory, "img")

    number_of_pages = pdf_to_image_converter(pdf_path, img_directory)

    for page in range(number_of_pages):
        img = f"page-{page}.png"
        vllm_extractor(img, current_directory)

    print(llm_summarizer("resp.txt", current_directory))


if __name__ == "__main__":
    typer.run(main)
