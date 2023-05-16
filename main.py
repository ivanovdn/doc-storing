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


def process_llm_response(llm_response):
    ans = {}
    ans["response"] = llm_response["result"]
    ans["source_documents"] = llm_response["source_documents"]
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


def return_document_metadata(query: str):
    pass


@st.cache_resource
def chat(_retriever, k_param):
    _retriever.search_kwargs = {"k": k_param}
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
    db, retriever = establish_db()
    qa_chain = chat(retriever, k_param=k_param)
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
        if mode == "Search":
            ans = db.similarity_search(query, k=k_param)
            for a in ans:
                dic = {}
                dic["page_content"] = a.page_content
                dic["source"] = a.metadata["source"]
                dic["page"] = a.metadata["page"] + 1
                st.write(dic)
        if mode == "Chat":
            llm_response = qa_chain(query)
            dic = process_llm_response(llm_response)
            st.write(dic)


if __name__ == "__main__":
    main()
