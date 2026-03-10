import os
import chromadb
from chromadb.utils import embedding_functions

class RAGRetriever:
	def __init__(
		self,
		k : int = 3,
		db_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/vector_store/")),
		collection_name: str = "biomedical_knowledge_base",
		embedding_model_name: str = "all-MiniLM-L6-v2"
	):	 
		"""
		Initializes the retriever to connect with the pre-built ChromaDB.
		By default the database is expected to live under the project's `data/vector_store` directory
		(it was previously stored on Google Drive).
		"""
		if not os.path.exists(db_path):
			raise FileNotFoundError(f"Database path not found: {db_path}. "
									f"Make sure the `data/vector_store` directory exists and contains the ChromaDB files.")
			
		# Store the number of top-k results to retrieve
		self.k = k
  
		# Match the exact embedding function used during database creation
		self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
			model_name=embedding_model_name
		)
		
		# Initialize the ChromaDB client pointing to the folder
		self.client = chromadb.PersistentClient(path=db_path)
		
		# Load the specific collection containing the data
		self.collection = self.client.get_collection(
			name=collection_name,
			embedding_function=self.embedding_function
		)

	def retrieve_context(self, query_text: str) -> str:
		"""
		Retrieves the top-k most relevant text chunks for the given sentence.
		"""
		# Query the vector database using k-NN
		# ChromaDB automatically handles the embedding of the query string here
		results = self.collection.query(
			query_texts=[query_text],
			n_results=self.k
		)

		# Extract and format the retrieved documents
		retrieved_chunks = results["documents"][0] if results["documents"] else []
		
		# Join the chunks into a single context string
		context_string = " | ".join(retrieved_chunks)
		
		return context_string

# --- Quick Test Block ---
if __name__ == "__main__":
	retriever = NERRetriever()
	
	# Test with a biomedical query
	sample_sentence = "The patient was diagnosed with severe hyperglycemia and requires immediate treatment."
	
	context = retriever.retrieve_context(sample_sentence, k=3)
	print(f"Retrieved Context:\n{context}")