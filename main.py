import re

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from config import config

config_dict = config.config()

st.title("Document storing example")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=config_dict["CHUNK_SIZE"],
    chunk_overlap=config_dict["CHUNK_OVERLAP"],
)


prompt_template = """Extract the key facts out of this text. Don't include opinions.


{text}

"""


PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])


refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "{query}"
)

refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "text", "query"],
    template=refine_template,
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
    if mode == "Search":
        ans["source_documents"] = response
    if mode == "Summary":
        ans["summary"] = response["output_text"]
    return ans


@st.cache_data
def load_docs(file_path: str):
    # st.info("`Reading doc ...`")
    if file_path.endswith(".pdf"):
        doc = PyPDFLoader(file_path)
        docs = doc.load_and_split(text_splitter=splitter)
    return docs


@st.cache_resource
def establish_db():
    # st.info("`Chroma DB creating...`")
    emb = SentenceTransformerEmbeddings(model_name=config_dict["MODEL_NAME"])
    db = Chroma(embedding_function=emb, persist_directory=config_dict["DB_PATH"])
    return db


@st.cache_resource
def establish_retriever(_db):
    retriever = _db.as_retriever()
    return retriever


@st.cache_resource
def define_llm():
    llm = ChatOpenAI(temperature=config_dict["TEMPERATURE"], request_timeout=120)
    return llm


@st.cache_resource
def chat(_retriever, _llm):
    qa_chain = RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=_retriever,
        return_source_documents=True,
    )
    return qa_chain


@st.cache_resource
def define_summarization_chain(_llm):
    chain = load_summarize_chain(
        _llm, chain_type="refine", question_prompt=PROMPT, refine_prompt=refine_prompt
    )
    return chain


@st.cache_data
def update_states_and_db(files: list, _db) -> None:
    for file_ in files:
        file_name = file_.name
        docs = load_docs(f"{config_dict['FILE_PATH']}/{file_name}")
        processed_docs = [preprocess(doc.page_content) for doc in docs]
        metadata = [doc.metadata for doc in docs]
        _db.add_texts(texts=processed_docs, metadatas=metadata)
        _db.persist()
        st.session_state.uploaded_files[file_name] = docs


def main():
    # Establish main components
    chromadb = establish_db()
    retriever = establish_retriever(chromadb)
    llm = define_llm()
    summarization_chain = define_summarization_chain(llm)
    qa_chain = chat(retriever, llm)

    k_param = 5
    retriever.search_kwargs = {"k": k_param}

    # Files
    files_list = st.file_uploader(
        "Upload Documents", type=["pdf", "jpg"], accept_multiple_files=True
    )

    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = {}

    if files_list:
        update_states_and_db(files_list, chromadb)
        st.write("Documents uploaded and processed.")

    # Sidebar
    st.sidebar.title("Menu")
    mode = st.sidebar.selectbox("Select Mode", ["Search", "Chat", "Summary"])

    if mode == "Summary":
        dock_to_summarize = st.sidebar.radio(
            "Pick dock to summarize", [file_.name for file_ in files_list]
        )
        print(st.session_state.uploaded_files[dock_to_summarize][:4])

    query = st.text_input(
        label=f"{mode}",
        value="",
        placeholder="Give each fact a number and keep them short sentences"
        if mode == "Summary"
        else "Enter query",
    )

    if len(query) > 0:
        if mode == "Search":
            response = retriever.get_relevant_documents(query=query)
        if mode == "Chat":
            response = qa_chain(query)
        if mode == "Summary":
            default_query = (
                f"Give each fact a number and keep them short sentences {query}"
            )
            response = summarization_chain(
                {"input_documents": summary_docs[:4], "query": default_query},
                return_only_outputs=True,
            )
        dic = process_response(response, mode)
        st.write(dic)


if __name__ == "__main__":
    main()
