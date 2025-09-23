import requests
import json


class BentoClient:
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url

    def encode_sentences(self, sentences: list[str]) -> dict:
        url = f"{self.base_url}/encode"
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        data = {"sentences": sentences}

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        return response.json()


if __name__ == "__main__":
    client = BentoClient()
    sentences = ["hello world"]
    try:
        result = client.encode_sentences(sentences)
        print(json.dumps(result, indent=2))
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
