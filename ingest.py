import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS

from dotenv import load_dotenv

load_dotenv()

def main():
    # Load document
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    persist_directory = os.environ.get("PERSIST_DIRECTORY")

    loader = PyPDFLoader("./COMP3221-W11-Lecture.pdf")
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Create the OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create the vectorstore and save to later use as the index
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

if __name__ == "__main__":
    main()