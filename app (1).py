import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Streamlit Setup ---
st.set_page_config(page_title="Chat with My PDF", page_icon="📚")
st.title("📚 Chat with Your PDF — AI RAG System")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("🔧 Settings")
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
    st.markdown("Once uploaded, the bot will read your document and let you chat with it.")

# --- Session state for memory ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file and openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Save uploaded PDF temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and process PDF
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    st.success("✅ PDF processed successfully! You can start chatting below.")

    # Display past messages
    for msg in st.session_state.messages:
        role = "🧑‍💻 You" if msg["role"] == "user" else "🤖 Assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])

    # Chat input for new question
    query = st.chat_input("Ask something about your PDF...")
    if query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("🧑‍💻 You"):
            st.markdown(query)

        # Generate answer
        with st.chat_message("🤖 Assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = qa.run(query)
                    st.markdown(answer)
                except Exception as e:
                    answer = f"❌ Error: {e}"
                    st.error(answer)

        # Save assistant reply
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Please upload a PDF and enter your OpenAI API key in the sidebar.")
