import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
import cohere
import gradio as gr
import os

# Initialize Sentence Transformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = 384  # Embedding dimension of 'all-MiniLM-L6-v2'

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Generate answer based on relevant text
def generate_answer(api_key, query, relevant_chunks):
    # Initialize the Cohere client using the provided API key
    cohere_client = cohere.Client(api_key)
    
    # Combine the relevant document chunks into a single context string
    context = " ".join(relevant_chunks)
    response = cohere_client.generate(prompt=f"Answer the question: {query} using the document's relevant context: {context}")
    return response.generations[0].text

# FAISS retrieval function
def retrieve_relevant_chunks_faiss(query, model, index, document_embeddings, document_chunks, top_k=5):
    query_embedding = model.encode([query])[0].astype(np.float32)
    query_embedding = query_embedding.reshape(1, -1)  # Reshape to be 2D for FAISS
    distances, indices = index.search(query_embedding, top_k)  # Search for top_k nearest neighbors
    relevant_chunks = [document_chunks[i] for i in indices[0]]  # Get relevant document chunks
    return relevant_chunks, distances

# Full document processing and query answering function
def QAbotdoc(api_key, document_path, query):
    document_text = extract_text_from_pdf(document_path)
    document_chunks = [document_text[i:i + 300] for i in range(0, len(document_text), 512)]
    
    # Encode document chunks into embeddings
    document_embeddings = model.encode(document_chunks)
    
    # Create FAISS index and add embeddings
    index = faiss.IndexFlatL2(embedding_dim)
    faiss_embeddings = np.array(document_embeddings).astype(np.float32)
    index.add(faiss_embeddings)
    
    # Retrieve relevant chunks and generate the answer
    relevant_chunks, distances = retrieve_relevant_chunks_faiss(query, model, index, document_embeddings, document_chunks)
    answer = generate_answer(api_key, query, relevant_chunks)
    
    return relevant_chunks, answer

# Gradio Interface
def gradio_interface(api_key, pdf_file, query):
    relevant_chunks, answer = QAbotdoc(api_key, pdf_file.name, query)
    return "\n\n".join(relevant_chunks), answer

# Launch Gradio interface
interface = gr.Interface(
    fn=gradio_interface, 
    inputs=[gr.Textbox(label="Enter Cohere API Key", type="password"),  # API Key input
            gr.File(label="Upload PDF"), 
            gr.Textbox(label="Enter your query")], 
    outputs=[gr.Textbox(label="Relevant Document Segments"), 
             gr.Textbox(label="Generated Answer")]
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=8080)
