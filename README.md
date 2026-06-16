# 📄 PDF RAG Chatbot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pdf-rag-chatbot-gikzsrgttvwauv4csigps7f.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> **Ask questions about your PDF documents and get instant, accurate answers using Groq's Llama 3.3 70B.**

<img width="1902" height="857" alt="Screenshot 2026-06-16 150804" src="https://github.com/user-attachments/assets/714d0008-4b8e-439e-aba3-bbe98353c224" />


GitHub :- [![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/Aikaksh-Singh-Routela/pdf-rag-chatbot)

LinkedIn :- [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-0A66C2?logo=linkedin)](https://www.linkedin.com/in/aikaksh-singh-routela/)


## 🚀 Features

- 📤 **Upload PDFs** - Process one or multiple PDF documents
- 🤖 **AI-Powered Q&A** - Get answers based ONLY on your documents
- ⚡ **Fast Inference** - Powered by Groq's Llama 3.3 70B (sub-5 second responses)
- 🔍 **RAG Architecture** - Retrieval Augmented Generation for accurate answers
- 💬 **Chat Interface** - Natural conversation with chat history
- 🔒 **Secure** - No data stored, API keys protected

  

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| [Streamlit](https://streamlit.io/) | Web framework & UI |
| [Groq](https://groq.com/) | Fast LLM inference (Llama 3.3 70B) |
| [LangChain](https://www.langchain.com/) | RAG pipeline orchestration |
| [FAISS](https://faiss.ai/) | Vector search & similarity |
| [HuggingFace](https://huggingface.co/) | Embeddings (all-MiniLM-L6-v2) |
| [PyPDF2](https://pypi.org/project/PyPDF2/) | PDF text extraction |

## 📦 Installation

### Prerequisites
- Python 3.11+
- Groq API key ([Get free key](https://console.groq.com))

### Setup
```bash
# Clone the repository
git clone https://github.com/Aikaksh-Singh-Routela/pdf-rag-chatbot.git
cd pdf-rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
🎯 Usage
Enter your Groq API key in the sidebar

Upload PDF documents (one or multiple)

Click "Process Documents" - wait ~30 seconds

Ask questions about your documents in the chat

Get instant answers based on the document content

Example Questions
"What is this document about?"

"Summarize the main points"

"What are the key findings?"

"List the important dates"

🌐 Live Demo
Try the app here: https://pdf-rag-chatbot-gikzsrgttvwauv4csigps7f.streamlit.app

📸 Screenshots
Upload & Processing
https://via.placeholder.com/800x400.png?text=Upload+PDF+and+Process

Chat Interface
https://via.placeholder.com/800x400.png?text=Ask+Questions+and+Get+Answers

🔄 How It Works
text
User Uploads PDF
        ↓
PDF Text Extraction (PyPDF2)
        ↓
Text Chunking (RecursiveCharacterTextSplitter)
        ↓
Embeddings Generation (HuggingFace)
        ↓
Vector Store Creation (FAISS)
        ↓
User Question → Retriever (Top 3 chunks)
        ↓
LLM Generation (Groq Llama 3.3 70B)
        ↓
Response Based ONLY on Document
🚀 Deployment
This app is deployed on Streamlit Community Cloud.

Push code to GitHub

Connect to Streamlit Cloud

Add GROQ_API_KEY as secret

Deploy!

🤝 Contributing
Contributions are welcome! Feel free to:

Open issues

Submit pull requests

Suggest improvements

📄 License
MIT License - see LICENSE file for details.

👨‍💻 Author
Aikaksh Singh Routela
