from voyager import Index, Space
from bento_client import BentoClient

client = BentoClient()

index = Index(Space.Cosine, num_dimensions=1024)

sentences = ["hello world", "goodbye world"]
embeddings = client.encode_sentences(sentences)["embeddings"]
index = Index(Space.Cosine, num_dimensions=len(embeddings[0]))
id_to_text = {}

for s, e in zip(sentences, embeddings):
    idx = index.add_item(e)
    id_to_text[idx] = s

query = "see you later"
vec = client.encode_sentences([query])["embeddings"][0]
neighbors, distances = index.query(vec, k=2)

for n, d in zip(neighbors, distances):
    print(f"Match: {id_to_text[n]}  |  Distance: {d}")
