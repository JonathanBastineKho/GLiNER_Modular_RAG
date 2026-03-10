from src.rag.retriever import RAGRetriever
from src.models import GLiNERRagConcat

def run_ner_pipeline():
    print("Initializing RAG-Augmented NER Pipeline...")
    
    # 1. Initialize both independent modules
    retriever = RAGRetriever(k=3)
    ner_model = GLiNERRagConcat("urchade/gliner_large-v1")
    
    # Define your target labels for the biomedical domain
    target_labels = ["disease", "chemical", "gene", "cell type"]
    
    # Sample input text
    input_text = "A biguanide hypoglycemic agent used in the treatment of non-insulin-dependent diabetes mellitus not responding to dietary modification."
    
    print(f"\nProcessing Text: {input_text}")
    
    # 2. Retrieval Step
    print("\nRetrieving context from ChromaDB...")
    context_string = retriever.retrieve_context(input_text)
    print(f"Retrieved Context: {context_string[:150]}...") # Printing a snippet
    
    # 3. Inference Step
    # Here is where you pass the output of your retriever into your teammate's model
    print("\nExtracting entities...")
    entities = ner_model.predict_entities(
        text=input_text, 
        labels=target_labels, 
        context=context_string,
    )
    
    # 4. Output Results
    print("\n--- Extracted Entities ---")
    for entity in entities:
        print(f"Text: {entity['text']} | Label: {entity['label']} | Score: {entity['score']:.3f}")

if __name__ == "__main__":
    run_ner_pipeline()