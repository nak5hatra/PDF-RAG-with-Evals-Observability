import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") # type: ignore


class PDFIngestion:
    
    """
    PDFIngestion Handles Ingestion of PDF Documents:
    
        1. Load PDFs
        2. Split into Chunks
        3. Store in Vector Database
    """
    
    def __init__(self, persist_directory: str = "chroma_store") -> None:
        self.persist_directory: str = persist_directory
        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings()
        
    def load_pdf(self, file_path: List[str]) -> List[Document]:
        
        """Load multiple PDF Files."""
        
        docs: List[Document] = []
        for path in file_path:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        return docs
    
    def split_docs(self, docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        
        """ Split Documents into smaller chunks."""
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        return splitter.split_documents(documents=docs)
    
    def create_vectorstore(self, docs: List[Document]) -> Chroma:
        
        """ Create or update Chroma vector store."""
        
        vectordb = Chroma.from_documents( # type: ignore
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        vectordb.persist()
        return vectordb
    
# if __name__ == "__main__":
#     ingestion = PDFIngestion()
#     pdf = ["/media/r00t/Neuraligence/resume Project/PDF-RAG-with-Evals-Observability/budget_speech.pdf"]
#     docs = ingestion.load_pdf(file_path=pdf)
#     chunks = ingestion.split_docs(docs=docs)
#     vectordb = ingestion.create_vectorstore(docs=chunks)
#     print(f"PDF ingestion is completed! and data is stored in {ingestion.persist_directory}")