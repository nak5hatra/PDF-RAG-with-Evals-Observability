import os
import time
import logging
from langsmith import Client
from rag_chain import rag_chain

from dotenv import load_dotenv
load_dotenv()


##  Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

LANGCHAIN_API_KEY= os.getenv("LANGCHAIN_API_KEY") # type: ignore
client = Client(api_key=LANGCHAIN_API_KEY) if LANGCHAIN_API_KEY else None

def observability(question: str):
    
    start_time = time.time()
    
    result = rag_chain.invoke({ # type: ignore
        "question": question,
        "context": [],
        "answer": ""
    })
    
    end_time = time.time()
    
    latency = round(end_time- start_time, 2)
    
    ##  Logs
    logging.info(f"Question: {question}")
    logging.info(f"Answer: {result["answer"][:200]}...")
    logging.info(f"latency: {latency} sec")
    
    if client:
        client.create_run(
            name="PDF RAG PIPELINE",
            inputs={"question": question},
            outputs={"answer": result["answer"]},
            metadata= {"latency": latency},
            run_type="chain",
        
        ) # type: ignore
        logging.info("Logged to LangSmith.")
    
    return latency, result


if __name__ == "__main__":
    q = "Summarize the PDF in 3 bullet points."
    latency, output = observability(q)
    print("\nFinal Answer:", output["answer"])
    print(f"Latency: {latency} sec")