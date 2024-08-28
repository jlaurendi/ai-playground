# Need to run this first: `huggingface-cli login`

# from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct")
pipe(messages)


# # Load a pre-trained model and tokenizer from Hugging Face
# # model_name = "facebook/bart-large-cnn"
# # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# # tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Initialize the pipeline for generation
# # generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# # Sample documents (this would normally be a larger dataset)
# documents = [
#     "Our refund policy allows customers to return items within 30 days.",
#     "To reset your password, visit the account settings page.",
#     "Shipping times vary depending on your location, usually 5-7 business days.",
#     # Add more documents as needed
# ]

# # Generate embeddings for the documents
# document_embeddings = model.encode(documents)

# # Create FAISS index for efficient similarity search
# dimension = document_embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(document_embeddings)

# def retrieve_documents(query, k=3):
#     # Convert query to embedding
#     query_embedding = model.encode([query])
    
#     # Search for the top-k similar documents
#     distances, indices = index.search(query_embedding, k)
    
#     # Return the top-k documents
#     return [documents[i] for i in indices[0]]

# # Example usage
# query = "How can I reset my password?"
# relevant_docs = retrieve_documents(query)
# print(relevant_docs)


# def generate_response(query):
#     # Retrieve relevant documents
#     relevant_docs = retrieve_documents(query)
    
#     # Construct the prompt with the retrieved documents
#     prompt = "Based on the following documents, answer the customer's question: "
#     prompt += " ".join(relevant_docs) + "\n\n" + "Question: " + query
    
#     # Generate the response
#     response = generator(prompt, max_length=150, num_return_sequences=1)
    
#     return response[0]['generated_text']

# # Example query
# query = "How can I reset my password?"
# response = generate_response(query)
# print(response)

# def evaluate_responses(queries, expected_responses):
#     correct = 0
#     total = len(queries)
    
#     for query, expected in zip(queries, expected_responses):
#         generated = generate_response(query)
#         print(f"Query: {query}")
#         print(f"Generated Response: {generated}")
#         print(f"Expected Response: {expected}\n")
        
#         if expected in generated:  # Simplistic check
#             correct += 1
    
#     accuracy = correct / total
#     print(f"Accuracy: {accuracy * 100:.2f}%")

# # Example evaluation
# queries = ["What is your refund policy?", "How do I reset my password?"]
# expected_responses = [
#     "Our refund policy allows customers to return items within 30 days.",
#     "To reset your password, visit the account settings page."
# ]

# evaluate_responses(queries, expected_responses)

