import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class DocumentProcessor:
    def __init__(self, vector_store_path: str = "./chroma_db"):
        self.vector_store_path = vector_store_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        # Using a default HuggingFace embedding model as BGE-M3 needs specific setup
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = Chroma(
            collection_name="insurance_docs",
            embedding_function=self.embeddings,
            persist_directory=self.vector_store_path
        )

    def load_documents(self, file_path: str) -> list[Document]:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        print(f"Loaded {len(docs)} documents from {file_path}")
        return docs

    def split_documents(self, documents: list[Document]) -> list[Document]:
        all_splits = self.text_splitter.split_documents(documents)
        print(f"Split documents into {len(all_splits)} chunks.")
        return all_splits

    def create_and_store_embeddings(self, splits: list[Document]):
        print(f"Storing {len(splits)} document chunks in the vector store.")
        self.vector_store.add_documents(documents=splits)
        print("Documents successfully stored in ChromaDB.")

    def get_retriever(self):
        return self.vector_store.as_retriever()

    def process_pdf(self, file_path: str):
        documents = self.load_documents(file_path)
        splits = self.split_documents(documents)
        self.create_and_store_embeddings(splits)
        print("PDF processing complete.")

if __name__ == "__main__":
    # Example usage:
    # Ensure you have a PDF file in the data/ directory, e.g., nke-10k-2023.pdf
    pdf_file_path = "data/nke-10k-2023.pdf"
    processor = DocumentProcessor()
    processor.process_pdf(pdf_file_path)

    # You can then get a retriever to perform similarity searches
    retriever = processor.get_retriever()
    query = "What is the average track length of genres?" # Example query, needs actual content
    results = retriever.invoke(query)
    print("\nSimilarity search results:")
    for doc in results:
        print(f"- Content: {doc.page_content[:200]}...")
        print(f"  Source: {doc.metadata.get('source')} (page {doc.metadata.get('page')})")
