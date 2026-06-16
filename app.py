import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile

# Page configuration
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .stButton > button {
        width: 100%;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📚 PDF RAG Chatbot")
st.markdown("*Ask questions about your documents - powered by Groq's ultra-fast LLM*")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get your free API key from console.groq.com"
    )
    
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    
    st.markdown("---")
    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF documents"
    )
    
    if uploaded_files and st.button("🔄 Process Documents", type="primary"):
        with st.spinner("Processing documents... This takes 30-60 seconds"):
            try:
                # Extract text from PDFs
                all_text = ""
                for pdf_file in uploaded_files:
                    pdf_reader = PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        all_text += page.extract_text()
                
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(all_text)
                
                # Create embeddings (using free HuggingFace model)
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
                # Create vector store
                vector_store = FAISS.from_texts(
                    chunks, 
                    embeddings
                )
                
                # Save to session state
                st.session_state.vector_store = vector_store
                st.session_state.retriever = vector_store.as_retriever(
                    search_kwargs={"k": 3}
                )
                
                st.success(f"✅ Processed {len(uploaded_files)} PDF(s) successfully!")
                
            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 📝 Example Questions")
    st.info("""
    - What is this document about?
    - Summarize the main points
    - What are the key findings?
    - List the important dates
    """)

# Main chat interface
st.header("💬 Ask Questions About Your Documents")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Check if documents are loaded
    if st.session_state.vector_store is None:
        st.error("Please upload and process documents first!")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Initialize Groq LLM with WORKING model
                    llm = ChatGroq(
                        model="llama-3.3-70b-versatile",  # UPDATED - replaced deprecated mixtral
                        temperature=0.3,
                        groq_api_key=groq_api_key or os.environ.get("GROQ_API_KEY")
                    )
                    
                    # Create custom prompt template
                    prompt_template = """You are a helpful assistant answering questions based ONLY on the provided document context.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the context above
- If the answer isn't in the context, say "I cannot find this information in the uploaded documents"
- Be concise and specific
- Cite the relevant information from the context

Answer: """

                    PROMPT = PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )
                    
                    # Create RAG chain using LCEL
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)
                    
                    rag_chain = (
                        {
                            "context": st.session_state.retriever | format_docs,
                            "question": RunnablePassthrough()
                        }
                        | PROMPT
                        | llm
                        | StrOutputParser()
                    )
                    
                    # Generate answer
                    answer = rag_chain.invoke(prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    Powered by Groq's Llama 3.3 70B | Embeddings: sentence-transformers | Vector Store: FAISS
</div>
""", unsafe_allow_html=True)