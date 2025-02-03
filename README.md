# 📚 Chatgroq with Llama3 Demo 🚀
Welcome to the Chatgroq with Llama3 Demo! This interactive application allows you to query multiple PDF documents and get precise, context-aware answers powered by advanced AI models like Llama3, GroqAPI, and Google Generative AI. Here's an overview of how it all works and what exciting features it offers:

![Image](https://github.com/user-attachments/assets/be566b2e-5f5b-411b-9375-08e96d2e33e8)



# 📄 What Does This Application Do?

📁 Multiple PDFs in a Folder: Upload a collection of PDF files into the ./PDFS folder, and the app will process them all! Whether you need to search one document or many, this app has you covered.

🔍 Ask Questions: Input your question, and the app retrieves the most relevant parts of the documents, providing an accurate answer based on context.

💡 Cutting-Edge Technology: The app leverages Retrieval-Augmented Generation (RAG), prompt engineering, and AI embeddings to deliver reliable and context-specific results.




# 🧠 What is RAG (Retrieval-Augmented Generation)?

RAG combines document retrieval and generative AI to produce precise, context-aware answers:


1. Retrieve: The most relevant document sections are identified using vector embeddings.
   
2. Generate: The generative AI model uses these retrieved sections to craft an accurate, context-specific response.

   
This approach ensures the answers are grounded in the provided documents.


# ✍️ What is Prompt Engineering?

Prompt Engineering is the art of crafting precise instructions (prompts) to guide the AI in producing optimal responses. Here, the prompts ensure that:

1. The AI uses only the provided context to answer.

2. It delivers clear and relevant answers to the questions.


# 🔗 What is a Chain?

Chains are logical workflows that combine various components like:

• Document Loaders: For loading data (e.g., PDFs).

• Text Splitters: To break content into manageable chunks.

• Vector Stores: To store embeddings for retrieval.

• LLMs: To generate responses.

In this app, we use the LangChain framework to create:

1. Document Combination Chain: For combining documents and prompting the AI.

2. Retrieval Chain: For retrieving the most relevant chunks from the vector store.

# 🌟 How GroqAPI Enhances This App

The GroqAPI connects to the Llama3 model, a high-performance large language model. Its integration ensures:

1. Efficient handling of large prompts and contexts.

2. Accurate and responsive answers.

3. Seamless integration with other AI components.




# ⚙️ How It Works

### 1. PDF Ingestion:

• PDFs are loaded using the PyPDFDirectoryLoader.

• Text from the PDFs is split into smaller chunks for embedding.

### 2. Vector Embeddings:

• Google's Generative AI creates embeddings, representing the document text in numerical form.

• These embeddings are stored in a vector database powered by FAISS.

### 3. Ask Your Question:

• Input your question in the text box.

• The app retrieves relevant document sections and generates a precise answer using Llama3.

### 4. Document Similarity Search:

• View the most relevant sections from the documents in the similarity search panel.



# 🛠️ Key Features

• 📂 Handles Multiple PDFs: Query across multiple files effortlessly.

• ⚡ RAG Framework: Combines retrieval and generation for accurate answers.

• 🎯 Precision with Prompt Engineering: Guides the AI for reliable responses.

• 🌍 Advanced AI Models: Uses GroqAPI, Llama3, and Google Generative AI.



# 🏁 Try It Now!

1. Place your PDFs in the ./PDFS folder.

2. Click "Documents Embedding" to initialize the vector store.

3. Enter your question and enjoy AI-powered insights!

🎉 Dive into the future of AI-enhanced document querying! 🧐



# Demo 📽

Below is a demonstration of how the application works:

![Demo of the Application](https://github.com/Abdelrahman-Amen/Multi_PDF_RAG_with_Groq/blob/main/Demo.gif)
