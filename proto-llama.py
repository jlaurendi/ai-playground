import json
# from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import requests

# model_name = 'meta-llama/Meta-Llama-3.1-8B'
# model_name = 'mlx-community/Phi-3-mini-4k-instruct-4bit'

# tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
# model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

documents = []
document_urls = [
    "https://datatracker.ietf.org/doc/html/rfc5322",
    "https://www.emailonacid.com/blog/article/email-development/email-development-best-practices-2/"
]
for url in document_urls:
    response = requests.get(url)
    documents.append(response.text)

embeddings = embedder.encode(documents)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

f = open('email.txt')
email_string = f.read()
f.close()
query = f"How can I optimize the SEO of my email? Here's my email: {email_string}"
query_embedding = embedder.encode([query])
distances, index = index.search(query_embedding, k=1)
retrieved_docs = [documents[i] for i in index[0]]

prompt = f"{query} Use these websites for additional context: {' '.join(retrieved_docs)}\n\n Answer:"
# inputs = tokenizer(prompt, return_tensors='pt')
# output = model.generate(**inputs, max_new_tokens=150)
model_url = "http://localhost:11434/api/generate"
data = {
  "model": "llama3.1",
  "prompt": prompt
}

response = requests.post(model_url, json=data, stream=True)
combined_response = ""
for chunk in response.iter_content(chunk_size=None):
    if chunk:
        decoded_chunk = chunk.decode('utf-8')
        json_data = json.loads(decoded_chunk)
        combined_response += json_data.get('response', '')    

# response = tokenizer.decode(output[0], skip_special_tokens=True)
print(combined_response)