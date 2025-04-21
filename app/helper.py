from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
import os

COLLECTION_NAME = "trafic_collection"
VECTOR_STORE_DIR = "./traffic_db"

def initialize_db(data_dir, embedding):
    """
    Initialize the database by loading documents from JSON files and creating a vector store.
    """
    

    # Initialize vector store
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=VECTOR_STORE_DIR
    )

    if len(vector_store.get()["documents"]) != 0:
        print("Database already initialized.")
        return vector_store

    documents = []
    data_folder = data_dir
    if not os.path.exists(data_folder):
        print(f"Data folder {data_folder} does not exist.")
        return vector_store
    else:
        print(f"Data folder {data_folder} exists.")

    for file_name in os.listdir(data_folder):
        if file_name.endswith(".json"):
            print(f"Processing file: {file_name}")
            loader = JSONLoader(
                file_path=os.path.join(data_folder, file_name),
                jq_schema=".[]",
                text_content=False
            )
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["source"] = file_name

            documents.extend(loaded_docs)
            print(f"Loaded {len(loaded_docs)} documents from {file_name}.")

    vector_store.add_documents(documents)
    print("Database initialized and documents loaded into the vector store.")
    return vector_store
