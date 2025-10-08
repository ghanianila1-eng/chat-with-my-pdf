import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Streamlit app setup
st.set_page_config(page_title="Chat with My PDF", page_icon="📚")
st.title("📚 Chat with Your PDF — AI RAG System")

# --- User Inputs ---
openai_api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")
uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])

# --- Main Logic ---
if uploaded_file and openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Save uploaded PDF temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF using LangChain's community loader
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    # Split text into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    # Create vector embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Create Retrieval-based QA system
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    st.success("✅ PDF processed successfully! Ask your question below:")

    # --- User Query ---
    query = st.text_input("💬 Ask a question about your PDF:")
    if query:
        with st.spinner("🤔 Thinking..."):
            try:
                answer = qa.run(query)
                st.write("💡 **Answer:**", answer)
            except Exception as e:
                st.error(f"❌ Error: {e}")
else:
    st.info("👆 Please upload a PDF and enter your OpenAI API key to begin.")

