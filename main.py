import re

import streamlit as st
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from langchain.chains import (
    ConversationalRetrievalChain,
    ConversationChain,
    RetrievalQA,
)
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from config import config
from prompt_templates import default_summary_query, prompt_template, refine_template

config_dict = config.config()

st.title("Document storing example")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=config_dict["CHUNK_SIZE"],
    chunk_overlap=config_dict["CHUNK_OVERLAP"],
)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

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
        ans["summary"] = response
    if mode == "OCR":
        ans["OCR"] = response
    if mode == "Conversation":
        ans["response"] = response
    return ans


@st.cache_data
def load_docs(file_path: str):
    # st.info("`Reading doc ...`")
    if file_path.endswith(".pdf"):
        doc = PyPDFLoader(file_path)
        docs = doc.load_and_split(text_splitter=splitter)
    else:
        single_img_doc = DocumentFile.from_images(file_path)
        model = ocr_predictor(
            det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True
        )
        result = model(single_img_doc)
        docs = result.export()
    return docs


def preprocess_ocr(docs):
    text = ""
    for block in docs["pages"][0]["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                text += word["value"] + " "
    return text


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
def update_states_and_db(file_: str, _db) -> None:
    file_name = file_.name
    docs = load_docs(f"{config_dict['FILE_PATH']}/{file_name}")
    if file_name.endswith(".pdf"):
        processed_docs = [preprocess(doc.page_content) for doc in docs]
        metadata = [doc.metadata for doc in docs]
        st.session_state.uploaded_files[file_name] = docs
    else:
        processed_docs = [preprocess_ocr(docs)]
        metadata = [{"source": file_name, "page": 1}]
        st.session_state.uploaded_files[file_name] = processed_docs
    _db.add_texts(texts=processed_docs, metadatas=metadata)
    _db.persist()


@st.cache_data
def summary_summary(_summary_chain, _docs):
    response = _summary_chain(
        {"input_documents": _docs, "query": default_summary_query},
        return_only_outputs=True,
    )
    return response


@st.cache_resource
def establish_conversation(_llm):
    window_memory = ConversationBufferWindowMemory(k=3)
    conversation = ConversationChain(memory=window_memory, llm=_llm, verbose=True)
    return conversation


def main():
    # Establish main components
    chromadb = establish_db()
    retriever = establish_retriever(chromadb)
    llm = define_llm()
    summarization_chain = define_summarization_chain(llm)
    qa_chain = chat(retriever, llm)
    conversation = establish_conversation(llm)

    k_param = 1
    retriever.search_kwargs = {"k": k_param}

    # Files
    files_list = st.file_uploader(
        "Upload Documents", type=["pdf", "jpg"], accept_multiple_files=True
    )

    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = {}
    if "summary" not in st.session_state:
        st.session_state["summary"] = {}

    if files_list:
        for file_ in files_list:
            update_states_and_db(file_, chromadb)
        st.write("Documents uploaded and processed.")

    # Sidebar
    st.sidebar.title("Menu")
    task = st.sidebar.radio("Pick file type", ["PDF", "JPG"])

    if task == "PDF":
        mode = st.sidebar.selectbox("Select Mode", ["Search", "Chat", "Summary"])
    if task == "JPG":
        mode = st.sidebar.selectbox("Select Mode", ["OCR", "Conversation"])

    if mode == "Summary":
        dock_to_summarize = st.sidebar.radio(
            "Pick dock to summarize",
            [file_.name for file_ in files_list if file_.name.endswith("pdf")],
        )
        summary_docs = st.session_state.uploaded_files[dock_to_summarize][:5]

        response = summary_summary(summarization_chain, summary_docs)
        summary = process_response(response, mode)
        st.session_state.summary[dock_to_summarize] = summary
        st.write(st.session_state.summary[dock_to_summarize])

    if task == "JPG":
        dock_to_summarize = st.sidebar.radio(
            "Pick dock to ocr",
            [file_.name for file_ in files_list if file_.name.endswith("jpg")],
        )
        ocr = st.session_state.uploaded_files[dock_to_summarize]
        if mode == "Conversation":
            st.write(ocr)

    query = st.text_input(
        label=f"{mode}",
        value="" if mode != "OCR" else "Extract",
        placeholder="Enter query",
    )

    if len(query) > 0:
        if mode == "Search":
            response = retriever.get_relevant_documents(query=query)
        if mode == "Chat":
            response = qa_chain(query)
        if mode == "OCR":
            response = ocr
        if mode == "Summary":
            response = conversation.predict(
                input=f"{query} {st.session_state.summary[dock_to_summarize]}"
            )
        if mode == "Conversation":
            response = conversation.predict(input=f"{query}")
        dic = process_response(response, mode)
        st.write(dic)


if __name__ == "__main__":
    main()
