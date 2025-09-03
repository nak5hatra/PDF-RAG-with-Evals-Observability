from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from .retriever import PDFRetriever

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") # type: ignore


class RAGState(TypedDict):
    question: str
    context: List[str]
    answer: str


##  Define retriever and LLM model
retriever = PDFRetriever()
llm = ChatOpenAI(model="gpt-5", temperature=0)



def retrieve_docs(state: RAGState) -> RAGState:

    docs = retriever.query(state["question"])
    context = [d.page_content for d in docs]

    return {
        "question": state["question"],
        "context": context,
        "answer": ""
    }

def generate_answer(state:RAGState) -> RAGState:
    
    prompt = ChatPromptTemplate.from_messages( # type: ignore
        messages=[
            ("system", "You are a helpful assistant. Please use the provided context to answer. if you don't know the answer just replay with Sorry, I don't have information about it."),
            ("human", "Question: {question}\n\nContext:\n{context}")
        ]
    )
    chain = prompt | llm # type: ignore
    
    response = chain.invoke( # type: ignore
        {
            "question": state["question"],
            "context": "\n\n".join(state["context"])
        }
    )
    return {
        "question": state["question"],
        "context": state["context"],
        "answer": response.content # type: ignore
    }

##  LangGraph Pipeline

graph = StateGraph(RAGState)
graph.add_node("retriever", retrieve_docs) # type: ignore
graph.add_node("generator", generate_answer) # type: ignore

graph.set_entry_point("retriever")
graph.add_edge("retriever", "generator")
graph.add_edge("generator", END)

rag_chain = graph.compile() # type: ignore

# if __name__ == "__main__":
#     question = "name of the person presenting the document?"
    
#     final_state = rag_chain.invoke( # type: ignore
#         input={
#             "question": question,
#             "context": [],
#             "answer": "",
#         }
#     )
#     print("Final Answer:")
#     print(final_state["answer"])