from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") # type: ignore

class PDFRetriever:
    
    """
    PDFRetriever Handles retrieval of documents from Chroma vector store.
    """
    
    def __init__(self, persist_directory: str = "chroma_store") -> None:
        self.persist_directory: str = persist_directory
        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings()
        self.vectordb: Chroma = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embeddings
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={'k': 3})
    
    def query(self, question: str):
        
        """Retrieve top-k documents for a query."""
        
        return self.retriever.invoke(input=question)
    
# if __name__ == "__main__":
#     retriever = PDFRetriever()
#     results = retriever.query("What's Gyan Bharatam")
#     print("Retrieved Results:")
#     for i, res in enumerate(results, start=1):
#         print(f"{i}: {res.page_content[:200]} \n\n")