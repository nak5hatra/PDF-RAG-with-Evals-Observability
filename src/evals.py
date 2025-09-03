from ragas import evaluate, EvaluationDataset # type: ignore
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from rag_chain import rag_chain

def evaluate_rag(samples: list): # type: ignore
    """Evaluates RAG pipeline using RAGS metrics

    Args:
        samples (list): list of dict with keys
        - "question": str
        - "ground_truth": str (expected answer)
    """
    predictions = []
    contexts = []
    
    for s in samples: # type: ignore
        result = rag_chain.invoke( # type: ignore
            {
                "question": s["question"],
                "context": [],
                "answer": "",
            }
        )
        predictions.append(result["answer"]) # type: ignore
        contexts.append(result["context"] if isinstance(result["context"], list) else [result["context"]]) # type: ignore
        
    dataset = EvaluationDataset.from_list({ # type: ignore
    "question": [s["question"] for s in samples], # type: ignore
    "answer": predictions,
    "contexts": contexts,
    "ground_truth": [s["ground_truth"] for s in samples], # type: ignore
    })

    scores = evaluate( # type: ignore
        dataset=dataset, # type: ignore
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )
    
    return scores

if __name__ == "__main__":
    test_samples = [
        {
            "question": "what is the main topic of the document?",
            "ground_truth": "BUDGET 2025-2026",
        },
        {
            "question": "who is the author of this document?",
            "ground_truth": "NIRMALA SITHARAMAN",
        }
    ]
    
    result = evaluate_rag(test_samples)
    print("RAG Evaluation Results\n")
    print(result)