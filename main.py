import re

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from config import config

config_dict = config.config()

st.title("Document storing example")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=config_dict["CHUNK_SIZE"],
    chunk_overlap=config_dict["CHUNK_OVERLAP"],
)


def preprocess(text: str) -> str:
    text = text.replace("\n", "")
    text = re.sub("\s+", " ", text)
    return text


def process_response(response, mode) -> dict:
    ans = {}
    if mode == "Chat":
        ans["response"] = response["result"]
        ans["source_documents"] = response["source_documents"]
    else:
        # ans["page_content"] = [resp.page_content for resp in response]
        # ans["source"] = [resp.metadata["source"] for resp in response]
        # ans["page"] = [resp.metadata["page"] + 1 for resp in response]
        ans["source_documents"] = response
    return ans


@st.cache_resource
def establish_db():
    st.info("`Chroma DB creating...`")
    emb = SentenceTransformerEmbeddings(model_name=config_dict["MODEL_NAME"])
    db = Chroma(embedding_function=emb, persist_directory=config_dict["DB_PATH"])
    retriever = db.as_retriever()
    return db, retriever


@st.cache_data
def load_docs(file_path: str):
    st.info("`Reading doc ...`")
    doc = PyPDFLoader(file_path)
    docs = doc.load_and_split(text_splitter=splitter)
    processed_docs = [preprocess(doc.page_content) for doc in docs]
    metadata = [doc.metadata for doc in docs]
    return processed_docs, metadata


@st.cache_resource
def chat(_retriever):
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=config_dict["TEMPERATURE"]),
        chain_type="stuff",
        retriever=_retriever,
        return_source_documents=True,
    )
    return qa_chain


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
    k_param = st.sidebar.selectbox("Select number of docs to return", [2, 3, 4, 5])

    # DB
    chromadb, retriever = establish_db()
    retriever.search_kwargs = {"k": k_param}
    qa_chain = chat(retriever)
    # Files
    files = st.file_uploader(
        "Upload Documents", type=["pdf"], accept_multiple_files=True
    )

    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    if files:
        update_states_and_db(files, chromadb)
        st.write("Documents uploaded and processed.")

    query = st.text_input(
        label=f"{mode}",
        value="",
        placeholder="Enter query",
    )

    if len(query) > 0:
        if mode == "Search":
            response = retriever.get_relevant_documents(query=query)
        if mode == "Chat":
            response = qa_chain(query)
        dic = process_response(response, mode)
        st.write(dic)


if __name__ == "__main__":
    main()
