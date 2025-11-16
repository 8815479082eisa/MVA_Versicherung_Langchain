from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", persist_directory="./chroma_db"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, add_start_index=True
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.persist_directory = persist_directory
        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self):
        # Initialize Chroma with the embedding function
        return Chroma(
            collection_name="insurance_documents",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def load_documents(self, file_path: str) -> list[Document]:
        """Loads documents from a given file path."""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        print(f"Loaded {len(docs)} documents from {file_path}")
        return docs

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Splits documents into smaller chunks."""
        splits = self.text_splitter.split_documents(documents)
        print(f"Split documents into {len(splits)} chunks.")
        return splits

    def index_documents(self, splits: list[Document]):
        """Indexes document chunks into the vector store."""
        print(f"Indexing {len(splits)} document chunks into the vector store...")
        self.vector_store.add_documents(documents=splits)
        print("Document indexing complete.")

    def get_retriever(self, search_type="similarity", k=5):
        """Returns a retriever from the vector store."""
        return self.vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})

if __name__ == "__main__":
    # Example Usage:
    # Create a dummy PDF for testing if one doesn't exist
    example_pdf_path = "example.pdf"
    if not os.path.exists(example_pdf_path):
        print(f"Creating a dummy PDF at {example_pdf_path} for demonstration.")
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(example_pdf_path)
        c.drawString(100, 750, "This is a sample insurance policy document.")
        c.drawString(100, 730, "It covers damages due to fire and theft.")
        c.drawString(100, 710, "The deductible for fire damage is $500.")
        c.drawString(100, 690, "The deductible for theft damage is $250.")
        c.drawString(100, 670, "Policy number: INS-PY-2023-001.")
        c.drawString(100, 650, "Coverage is valid for the year 2023.")
        c.save()

    processor = DocumentProcessor()
    documents = processor.load_documents(example_pdf_path)
    splits = processor.split_documents(documents)
    processor.index_documents(splits)

    # You can now use the retriever to search
    retriever = processor.get_retriever(k=2)
    query = "What is the deductible for fire damage?"
    results = retriever.invoke(query)
    print(f"\nRetrieval results for query: '{query}'")
    for doc in results:
        print(f"Content: {doc.page_content[:150]}...")
        print(f"Metadata: {doc.metadata}\n")
