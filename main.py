import re

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from config import config

config_dict = config.config()

st.title("Document storing example")

splitter = CharacterTextSplitter(
    separator=config_dict["SEPARATOR"],
    chunk_size=config_dict["CHUNK_SIZE"],
    chunk_overlap=config_dict["CHUNK_OVERLAP"],
)


def preprocess(text: str) -> str:
    text = text.replace("\n", "")
    text = re.sub("\s+", " ", text)
    return text


@st.cache_resource
def establish_db():
    st.info("`Chroma DB creating...`")
    emb = SentenceTransformerEmbeddings(model_name=config_dict["MODEL_NAME"])
    db = Chroma(embedding_function=emb, persist_directory=config_dict["DB_PATH"])
    return db


@st.cache_data
def load_docs(file_path: str):
    st.info("`Reading doc ...`")
    doc = PyPDFLoader(file_path)
    docs = doc.load_and_split(text_splitter=splitter)
    processed_docs = [preprocess(doc.page_content) for doc in docs]
    metadata = [doc.metadata for doc in docs]
    return processed_docs, metadata


def return_document_metadata(query: str):
    pass


def chat():
    pass


def update_states_and_db(files: list, db) -> None:
    for file_ in files:
        if file_.name not in st.session_state["uploaded_files"]:
            st.session_state.uploaded_files.append(file_.name)
            docs, metadata = load_docs(f"{config_dict['FILE_PATH']}/{file_.name}")
            db.add_texts(texts=docs, metadatas=metadata)
            db.persist()


def main():
    # Sidebar
    st.sidebar.title("Menu")
    mode = st.sidebar.radio("Select Mode", ["Search", "Chat"])

    # DB
    db = establish_db()
    # Files
    files = st.file_uploader(
        "Upload Documents", type=["pdf"], accept_multiple_files=True
    )

    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    if files:
        update_states_and_db(files, db)
        st.write("Documents uploaded and processed.")

    query = st.text_input(
        label=f"{mode}",
        value="",
        placeholder="Enter query",
    )

    if len(query) > 0:
        ans = db.similarity_search(query, k=2)
        for a in ans:
            dic = {}
            dic["page_content"] = a.page_content
            dic["source"] = a.metadata["source"]
            dic["page"] = a.metadata["page"] + 1
            st.write(dic)


if __name__ == "__main__":
    main()
