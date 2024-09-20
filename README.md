# Retrieval-Augmented Generation (RAG) Based Question Answering (QA) System

This repository contains the implementation of a **Retrieval-Augmented Generation (RAG)** model for a document-based **Question Answering (QA) System**. The system utilizes **FAISS** for efficient document retrieval and **Cohere API** for generating contextually accurate answers. The goal of this project is to allow users to upload documents (PDFs), ask questions, and receive accurate answers by retrieving relevant document chunks and generating a coherent response.

## Features
- **PDF Document Upload**: Users can upload PDF files and interact with the content.
- **Efficient Retrieval**: Uses **FAISS** to retrieve the most relevant document segments.
- **Answer Generation**: Uses **Cohere API** to generate fluent and contextually accurate answers.
- **User-Friendly Interface**: Powered by **Gradio** for an easy-to-use, real-time interface.

## Google Collab Notebooks and Documentation
- [RAG QA Bot](https://colab.research.google.com/drive/1DyXf1VKLAy3DyqB0STkvhbHTzAeTpGZE)
- [Integration with UI and Deployment](https://colab.research.google.com/drive/1NrnVZIBMROlVMVbGLanN2YxUabyjsYTd)
- [Usage Document ](https://docs.google.com/document/d/17f-3eyEEg4DHwMUo1TBQw1YQc4a1IUzN_hB92K1esH4)

## Installation

### 1. Clone the Repository

git clone https://github.com/reddy24043819/RAG_QAbot.git

### Introduction
  This project demonstrates the development of a Retrieval-Augmented Generation (RAG) model for building a Question Answering (QA) system using a combination of FAISS for efficient document retrieval and Cohere API for natural language generation. The system is built to process user-uploaded documents, extract relevant chunks of information based on a user query, and generate a coherent answer. A great thanks to Cohere API, Gradio and Docker.

### **Pipeline Overview**

  The system consists of two primary components:
Document Embedding and Retrieval (FAISS): The system embeds a document into a vector space and uses FAISS (Facebook AI Similarity Search) to retrieve the most relevant chunks of the document based on a user query.
Answer Generation (Cohere API): The relevant document chunks are passed to the Cohere API, which generates a coherent and contextually accurate answer based on the retrieved information.
The Gradio interface is used for user interaction, where users can upload documents and ask questions.

### **Pipeline Breakdown**

1. Document Preprocessing
  The first step in the pipeline is extracting the content from the uploaded document. In this implementation, the documents are expected to be PDF files.
PDF Extraction: We use the pdfplumber library to extract text from the uploaded PDF file.

2. Chunking the Document
  Once the document's text is extracted, it is split into smaller chunks for processing. This is important to ensure that:
The chunks are small enough to be encoded efficiently.
The language model handles the chunks better, without hitting token limits.
Chunking Logic: The document is split into 300-character chunks with a sliding window of 512 characters to provide context overlap between adjacent chunks.

3. Document Embedding
  To compare the document content with a user query, both the document chunks and the query must be transformed into a vector representation. 
We use the Sentence-Transformers library to generate embeddings for each document chunk.
Sentence-Transformers  is designed to create high-quality sentence and document embeddings. The model we are using, all-MiniLM-L6-v2, is a small, efficient model that provides a good balance between speed and accuracy.

**Embedding Process:**
    Document Chunk Embeddings: After splitting the document into smaller chunks, each chunk is encoded into a 384-dimensional vector using the Sentence-Transformer model.
  Query Embeddings: When a user inputs a query, the same transformer model is used to encode the query into a vector that can be compared with the document chunk embeddings.
  This encoding allows the system to understand both the content of the document and the query at a semantic level, ensuring that even if the words differ, the meanings are captured effectively.

**4. FAISS Index Creation**

  Once the document chunks are embedded, we use FAISS (Facebook AI Similarity Search) to build an index. FAISS is a library optimized for similarity search, which allows us to efficiently find     the most relevant document chunks based on the query.
  Index Creation: We create a FAISS index with the document embeddings.
  Similarity Search: For each query, we use FAISS to find the top-k most relevant document chunks based on vector similarity. 
  Here top k was originally set to 2 , found the model had very little context to get information from , so later it was set to 5.

**5. Answer Generation (Cohere API)**

  After retrieving the most relevant document chunks through FAISS, the next step is to generate a clear and coherent answer based on the user’s query. To achieve this, we leverage the Cohere API, which is designed for high-quality text generation.
  The Cohere API provides powerful language models that excel at understanding context and generating human-like responses. 
  We take advantage of this by passing the retrieved document chunks, combined with the user’s query, as a prompt to the API. The model then generates a well-structured, contextually relevant answer that directly addresses the query.

**How It Works:**
  Contextual Input: The retrieved chunks (which contain the most relevant information from the document) are combined into a single block of text. This text, along with the user’s query, forms the prompt for the generative model.
  
  Query-Specific Response: The Cohere model uses this context to generate an answer that is tailored to the user’s query. This ensures that the response is not generic but instead provides detailed, context-aware information.
By using a combination of document context and the user query, the model can formulate answers that closely align with the content of the document while addressing the specific needs of the user.

**6. Testing the QA System**
    This phase involves evaluating how well the system retrieves relevant chunks, how accurate the generated answers are, and how the system handles various types of queries and document inputs.
    Testing Objectives:
    
  **Verify Retrieval Accuracy**: Ensure that the system retrieves the most relevant document chunks based on user queries.
    Evaluate Answer Quality: Test whether the answers generated by the Cohere API are coherent, contextually accurate, and informative. The generated answer should be evaluated based on the following criteria:
    
    Relevance: Does the answer address the query appropriately?
    Coherence: Is the answer easy to understand and well-structured?
    Accuracy: Is the information provided correct, and does it match the document's content?
    **Assess Performance**: Check the system's response time and performance, especially when handling multiple queries or large documents.
    **Test Edge Cases:** Ensure the system handles different edge cases, such as ambiguous queries, very long documents, or queries that have no clear answer in the document.

### User Interface using Gradio
**7. Gradio Interface**

  **Gradio is a Python library that enables you to quickly create a web interface for machine learning models. In this step, we will use Gradio to allow users to interact with the system by:**
    Uploading a PDF document.
    Typing a query related to the document.
    Receiving a response that includes the relevant document segments and a generated answer.
  
  This interface simplifies the testing process and provides a user-friendly way to validate the system.
 
  1. Define the Gradio Interface Function
      The main function that drives the Gradio interface will:
      Take the uploaded PDF and the user's query as inputs.
      Call the QAbotdoc function, which extracts text, processes embeddings, retrieves relevant chunks, and generates an answer.
      Return both the relevant document segments and the generated answer.
      The function returns:
      Relevant Document Segments: These are the most relevant chunks of the document that relate to the query.
      Generated Answer: This is the answer generated by the Cohere API based on the retrieved document segments.
     
  3. Define the Gradio Inputs and Outputs
      Next, define the inputs (PDF file and query) and the outputs (relevant document segments and generated answer) for the Gradio interface.
     
  5. Launch the Gradio Interface
      Now that the input and output components are defined, we can launch the Gradio interface. The Gradio app will open a link in the notebook output that allows you to interact with the system.


**Upload the pdf and enter your query in the space provided**

**Hit Submit to get the Generated Answer and Relevant Document Text it was deduced from.**

### Deployment

**8. Docker Depoloyment**

**This documentation explains how to deploy a Retrieval-Augmented Generation (RAG) model-based QA bot using Docker. The deployment allows the user to upload documents, ask questions, and get answers based on document retrieval and generation using the Cohere API.
Additionally, users will be prompted to enter their Cohere API key securely in the UI.**

Download and install Docker.

Cohere API Key: Sign up and get an API key from Cohere.

**Other required files** 

  1.**app.py**: The Main Application
  This file contains the logic for document processing, embedding, FAISS-based retrieval, and question-answer generation. It uses Gradio to create a web interface where users can enter the   Cohere API key, upload a PDF, and ask questions.
  
  2. **requirements.txt**: List of Dependencies
    Create a requirements.txt file to list the Python packages required for the app:
    faiss-cpu
    cohere
    sentence-transformers
    pdfplumber
    gradio
  3. **Dockerfile**: Docker Configuration
  
  This file defines the environment and steps needed to build the Docker image for the QA bot application.
  
**Project Structure**

  You need the following files in your project directory:

projectqa/
│
├── app.py              # Main Python application file
├── requirements.txt    # Python dependencies
└── Dockerfile          # Dockerfile to build the container

​​Building and Running the Docker Container

Step 1: Build the Docker Image

Open your terminal and navigate to the project directory where the Dockerfile, app.py, and requirements.txt files are located. Run the following command to build the Docker image:

    docker build -t projectqa .

This command builds the Docker image with the name projectqa.

Step 2: Run the Docker Container

Once the Docker image is built, run the container:

    docker run -p 8080:8080 projectqa

This command runs the Docker container and maps port 8080 of the container to port 8080 on your local machine.

*If port 8080 is in use, you can choose another port by changing the command, for example: -p 9090:8080.

Step 3: Access the Application

After running the container, open your browser and go to:

    http://localhost:8080

You will see the Gradio interface where you can:

**Enter your Cohere API key.
Upload a PDF document.
Input a query.**


#### The QA bot will retrieve relevant document segments and generate a response using the Cohere API.





