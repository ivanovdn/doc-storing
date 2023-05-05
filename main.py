import re

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

st.title("Document storing example")

FILE_PATH = "/Users/dmytro.ivanov/Projects/nlp/chat-pdf/data"
DB_PATH = "/Users/dmytro.ivanov/Projects/nlp/chat-pdf/data/db"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

splitter = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=100)


def preprocess(text: str) -> str:
    text = text.replace("\n", "")
    text = re.sub("\s+", " ", text)
    return text


@st.cache_resource
def establish_db():
    st.info("`Chroma DB creating...`")
    emb = SentenceTransformerEmbeddings(model_name=MODEL_NAME)
    db = Chroma(embedding_function=emb, persist_directory=DB_PATH)
    return db


@st.cache_data
def load_docs(file_path: str):
    st.info("`Reading doc ...`")
    doc = PyPDFLoader(file_path)
    docs = doc.load_and_split(text_splitter=splitter)
    docs = [preprocess(doc.page_content) for doc in docs]
    metadata = [doc.metadata for doc in docs]
    return docs, metadata


def return_document_metadata(query: str):
    pass


def update_states_and_db(files: list, db) -> None:
    for file_ in files:
        if file_.name not in st.session_state["uploaded_files"]:
            st.session_state.uploaded_files.append(file_.name)
            docs, metadata = load_docs(f"{FILE_PATH}/{file_.name}")
            db.add_texts(texts=docs, metadatas=metadata)
            db.persist()


def main():
    db = establish_db()
    files = st.file_uploader(
        "Upload Documents", type=["pdf"], accept_multiple_files=True
    )

    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    if files:
        update_states_and_db(files, db)
        print(st.session_state.uploaded_files)

        st.write("Documents uploaded and processed.")

    query = st.text_input(
        label="Enter query",
        value="",
        placeholder="Enter query",
        label_visibility="hidden",
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
