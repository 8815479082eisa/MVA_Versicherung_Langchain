import os
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_chroma import Chroma
from src.document_handling.document_processor import DocumentProcessor
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever


class HybridRetrievalPipeline:
    def __init__(self, vector_store: Chroma, documents: List[str], 
                 reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.vector_store_retriever = vector_store.as_retriever()
        self.bm25_retriever = BM25Retriever.from_texts(documents)
        self.ensemble_retriever = EnsembleRetriever(retrievers=[self.bm25_retriever, self.vector_store_retriever], weights=[0.5, 0.5])

        # Initialize Cross-Encoder Reranker
        self.compressor = CrossEncoderReranker(model_name=reranker_model_name, top_n=5)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.ensemble_retriever
        )

    def retrieve(self, query: str):
        return self.compression_retriever.invoke(query)


if __name__ == "__main__":
    # Example usage:
    # Initialize DocumentProcessor to load and process documents
    processor = DocumentProcessor()
    
    # To properly initialize BM25, we need the raw text content of all splits.
    # Let's re-process the PDF and get the splits.
    pdf_file_path = "data/nke-10k-2023.pdf"
    documents = processor.load_documents(pdf_file_path)
    all_splits = processor.split_documents(documents)
    processor.create_and_store_embeddings(all_splits) 
    
    # Extract page_content for BM25Retriever
    texts = [doc.page_content for doc in all_splits]

    hybrid_pipeline = HybridRetrievalPipeline(processor.vector_store, texts)

    query = "NIKE's revenue in 2023"
    results = hybrid_pipeline.retrieve(query)

    print(f"\nHybrid search results for query '{query}':")
    for doc in results:
        print(f"- Content: {doc.page_content[:200]}...")
        print(f"  Source: {doc.metadata.get('source')} (page {doc.metadata.get('page')})")
