import os
import base64
import pymupdf
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)


def convert_pdf_to_images(pdf_path: str, output_dir: str) -> int:
    """
    Convert PDF pages to images and save them in the specified output directory.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Directory to save the output images.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap()  # type: ignore
        image_path = os.path.join(output_dir, f"page-{page.number}.png")
        pix.save(image_path)

    return len(doc)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def llm_extract_slide_info(img: str):
    img_path = os.path.join(current_directory, "img", img)
    base64_image = encode_image(img_path)
    completion = client.chat.completions.create(
        model=os.getenv("VISION_MODEL_NAME", ""),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the content from the provided document as if you were reading it naturally.",
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


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_directory, "oop.pdf")
    img_directory = os.path.join(current_directory, "img")

    number_of_pages = convert_pdf_to_images(pdf_path, img_directory)

    for page in range(number_of_pages):
        img = f"page-{page}.png"
        llm_extract_slide_info(img)
