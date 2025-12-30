from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core import documents
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def load_pdf_dictionary(pdf_path: str) -> list[documents.Document]:
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def split_dictionary_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    split_docs = splitter.split_documents(docs)
    print(f"Loaded and split into {len(split_docs)} chunks.")
    return split_docs



def build_dictionary_vectorstore(docs, chroma_name):
    # embeddings = OllamaEmbeddings(model="llama3")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")


    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_atilf_en_dict",
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("Indexed into Chroma.")
    return vectordb

def load_dictionary_vectorstore(chroma_name):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
                                
    vectordb = Chroma(
        persist_directory=chroma_name,
        embedding_function=embeddings,
    )

    return vectordb



@tool(response_format="content_and_artifact")
def retrieve_dict(word: str): 
    """Retrieve Dictionary explanations for a word."""
    docs = retriever.invoke(word)
    serialized = "\n\n".join(doc.page_content for doc in docs)
    return serialized, docs



from langchain_ollama import ChatOllama
from langchain.agents import create_agent

def make_vector_db(pdf, chroma_name):
    # Load dictionary pdf
    raw_docs = load_pdf_dictionary(pdf)
    split_docs = split_dictionary_docs(raw_docs)
    # Build vector store
    vectordb = build_dictionary_vectorstore(split_docs, chroma_name)
    return vectordb


if __name__ == "__main__":
    
    PDF_PATH = "DECT_English_20141201.pdf"
    chroma_name = "chroma_atilf_en_dict"

    # Create vector store from PDF
    # vectordb = make_vector_db(PDF_PATH, chroma_name)

    # Or load existing vector store
    # vectordb = load_dictionary_vectorstore(chroma_name)
    # retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    
    # rag_agent = create_agent(
    # model=ChatOllama(model="llama3.1:8b", temperature=0.1),
    # tools=[retrieve_dict],
    # system_prompt="""
    # You are a helpful assistant specialized in historical linguistics and Old French.
    # your task is to identify important and ambiguous words in Old French text and provide accurate definitions and explanations for them.

    # You have access to a tool `retrieve_dict(word)` which returns relevant excerpts from a dictionary PDF.
    # for given phrase look up the definitions for ambiguous old french words and return summarized explanations. and different meanings if available.
    # make sure to cover all possible lemmas and meanings for each word.
    # """,
    # )





