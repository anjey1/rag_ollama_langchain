from flask import Flask, request
from langchain_community.llms.ollama import Ollama

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain.prompts import PromptTemplate


from langchain.text_splitter import RecursiveCharacterTextSplitter

import sqlite3

print(sqlite3.sqlite_version)

import logging

logging.basicConfig(
    filename="langchain_logs.log",  # The log file where logs will be written
    level=logging.INFO,  # The lowest level to capture (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log message format
)

lang_logger = logging.getLogger("my_logger")
lang_logger.info(f"Testing Logger... 1, 2 .... 1, 2 ....")

app = Flask(__name__)

folder_path = "db"

# Initialize the Ollama model with LangChain
cached_llm = Ollama(model="llama3")

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template()


@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query : {query}")

    # Use the invoke method to send a prompt
    response_answer = cached_llm.invoke(query)

    print(response_answer)

    response = {"answer": response_answer}
    return response


@app.route("/pdf", methods=["POST"])
def pdfPost():
    """Pdf Post Route\n

    Returns:
        json:
            { "chunks_len": 208, "doc_len": 111, "file_name": "alices-adventures-in-wonderland.pdf", "status": "Uploaded"}
    """
    print("Post /pdf called")
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename saved as: {file_name}")
    lang_logger.info(f"filename saved as: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len=: {len(docs)}")
    lang_logger.info(f"docs len=: {len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len=: {len(chunks)}")
    lang_logger.info(f"chunks len=: {len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )
    lang_logger.info(f"Vector Store Created")

    vector_store.persist()
    lang_logger.info(f"Vector Store Saved")

    response = {
        "status": "Uploaded",
        "file_name": file_name,
        "doc_len": len(docs),
        "chunks_len": len(chunks),
    }

    return response


@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query : {query}")

    # Use the invoke method to send a prompt
    # response_answer = cached_llm.invoke(query)
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print("Creating retriver")

    retriver = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    print("Creating chain")

    document_chain = create_stuff_documents_chain(cached_llm, raw_propmt)
    chain = create_retrieval_chain(retriver, document_chain)

    result = chain.invoke({"input": query})

    response_answer = {"answer": result}
    return response_answer


def start_app():
    app.run(host="0.0.0.0", port=7000, debug=True)


if __name__ == "__main__":
    start_app()
