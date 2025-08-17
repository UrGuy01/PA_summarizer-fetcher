import os
import streamlit as st
import pickle

from langchain_community.llms import Ollama
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Force CPU use
os.environ["OLLAMA_NO_GPU"] = "1"

llm = Ollama(model='mistral')

st.title("📰 News Research Tool")
st.sidebar.title("New Article Links")

file_path = 'faiss_stored_huggingface_embedds.pkl'

# Sidebar inputs for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URL")

main_placeholder = st.empty()

if process_url_clicked:
    main_placeholder.text("🔄 Loading data from URLs...")
    
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    
    main_placeholder.text("✂️ Splitting into chunks...")
    # Split data into manageable chunks
    re_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ' '],
        chunk_size=700,
        chunk_overlap=200
    )
    docs = re_splitter.split_documents(data)
    
    main_placeholder.text("🧠 Creating embeddings + FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)  
    
    # Save FAISS vector store
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)
    
    main_placeholder.text("✅ Data processed and FAISS index created!")

# Question input
query = st.text_input("Ask a question:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa_chain({'query': query})
        
        st.header("Answer")
        st.write(result['result'])
        
        # Display sources as clickable links
        sources = result.get("source_documents", [])
        if sources:
            st.subheader("Sources:")
            for doc in sources:
                # Most loaders store URL in doc.metadata['source']
                url = doc.metadata.get('source', None)
                if url:
                    st.markdown(f"[{url}]({url})")
