import os
import tempfile
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI

from langchain.chains import RetrievalQA


# -------------------------------------------------
# Load Environment Variables
# -------------------------------------------------

load_dotenv()

# -------------------------------------------------
# LangSmith Monitoring
# -------------------------------------------------

os.environ["LANGCHAIN_TRACING_V2"] = "true"

os.environ["LANGCHAIN_API_KEY"] = os.getenv(
    "LANGCHAIN_API_KEY",
    ""
)

os.environ["LANGCHAIN_PROJECT"] = os.getenv(
    "LANGCHAIN_PROJECT",
    "MultiSource-RAG"
)

# -------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Multi Source RAG",
    layout="wide"
)

st.title("📚 Multi-Source RAG App")
st.write(
    "Upload files and ask questions across documents."
)

# -------------------------------------------------
# IMPORTANT:
# ONLY THESE FILES WILL BE LOADED
# -------------------------------------------------

ALLOWED_FILES = {
    "NetApp.pdf",
    "RBI_amendment.txt"
}

# -------------------------------------------------
# Upload Files
# -------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload Documents",
    accept_multiple_files=True,
    type=["pdf", "txt"]
)

# -------------------------------------------------
# Document Loader
# -------------------------------------------------

def load_documents(files):

    all_docs = []

    for uploaded_file in files:

        filename = uploaded_file.name

        # -----------------------------------------
        # Ignore all other files
        # -----------------------------------------

        if filename not in ALLOWED_FILES:

            st.warning(
                f"Ignoring file: {filename}"
            )

            continue

        suffix = os.path.splitext(filename)[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        # -----------------------------------------
        # PDF Loader
        # -----------------------------------------

        if suffix.lower() == ".pdf":

            loader = PyPDFLoader(temp_path)

            docs = loader.load()

            for doc in docs:

                doc.metadata["source"] = filename

        # -----------------------------------------
        # TXT Loader
        # -----------------------------------------

        elif suffix.lower() == ".txt":

            loader = TextLoader(
                temp_path,
                encoding="utf-8"
            )

            docs = loader.load()

            for doc in docs:

                doc.metadata["source"] = filename
                doc.metadata["page"] = "N/A"

        else:

            continue

        all_docs.extend(docs)

    return all_docs

# -------------------------------------------------
# Create RAG Pipeline
# -------------------------------------------------

if uploaded_files:

    with st.spinner("Loading documents..."):

        documents = load_documents(
            uploaded_files
        )

        if len(documents) == 0:

            st.error(
                "No valid files uploaded."
            )

            st.stop()

        # -----------------------------------------
        # Text Splitting
        # -----------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(
            documents
        )

        # -----------------------------------------
        # Embedding Model
        # -----------------------------------------

        embeddings = HuggingFaceEmbeddings(
            model_name=(
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            )
        )

        # -----------------------------------------
        # FAISS Vector DB
        # -----------------------------------------

        vector_db = FAISS.from_documents(
            chunks,
            embeddings
        )

        # -----------------------------------------
        # Retriever
        # -----------------------------------------

        retriever = vector_db.as_retriever(
            search_kwargs={"k": 3}
        )

        # -----------------------------------------
        # LLM
        # -----------------------------------------

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        # -----------------------------------------
        # QA Chain
        # -----------------------------------------

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

    st.success(
        f"Loaded {len(chunks)} chunks successfully ✅"
    )

    # -------------------------------------------------
    # Question Box
    # -------------------------------------------------

    query = st.text_input(
        "Ask a question"
    )

    if query:

        with st.spinner(
            "Searching knowledge base..."
        ):

            result = qa_chain(
                {"query": query}
            )

            answer = result["result"]

            source_docs = result[
                "source_documents"
            ]

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Sources")

        displayed = set()

        for doc in source_docs:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "N/A"
            )

            key = f"{source}_{page}"

            if key not in displayed:

                st.write(
                    f"📄 File: {source} | Page: {page}"
                )

                displayed.add(key)

        st.subheader(
            "Retrieved Chunks"
        )

        for i, doc in enumerate(
                source_docs,
                start=1
        ):
            with st.expander(
                f"Chunk {i}"
            ):
                st.write(
                    doc.page_content
                )