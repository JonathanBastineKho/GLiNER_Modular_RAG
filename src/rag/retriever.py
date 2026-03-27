import os
import chromadb
from chromadb.utils import embedding_functions
import spacy
from rank_bm25 import BM25Okapi
import pickle

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

		print("Loading SciSpacy biomedical model...")
		try:
			self.nlp = spacy.load("en_core_sci_sm")
		except OSError:
			raise OSError("Could not load en_core_sci_sm. Please install it via the provided URL.")
		
		cache_path = os.path.join(db_path, "bm25_index.pkl")

		if os.path.exists(cache_path):
			print("Loading BM25 Index from cache...")
			with open(cache_path, "rb") as f:
				self.bm25, self.corpus_docs = pickle.load(f)
		else:
			all_data = self.collection.get(include=['documents'])
			self.corpus_docs = all_data['documents']
			
			print(f"Building BM25 Index for {len(self.corpus_docs)} chunks...")
			# Simple whitespace tokenization for BM25
			tokenized_corpus = [doc.split(" ") for doc in self.corpus_docs]
			self.bm25 = BM25Okapi(tokenized_corpus)
			with open(cache_path, "wb") as f:
				pickle.dump((self.bm25, self.corpus_docs), f)
			print("Hybrid Retriever Initialization Complete.")
	
	def extract_keywords(self, sentence_text: str) -> str:
		"""Uses SciSpacy to drop stop words/verbs and keep only biomedical noun chunks."""
		doc = self.nlp(sentence_text)
		keywords = [chunk.text for chunk in doc.ents]
		
		# Fallback to original text if the model finds no specific chunks
		if not keywords:
			return sentence_text
			
		return " ".join(keywords)
	
	def retrieve_context(self, query_text: str) -> str:
		# 1. Clean the query
		clean_query = self.extract_keywords(query_text)
		
		# 2. Dense Retrieval (Semantic Level via PubMedBERT)
		# We retrieve more than k to allow RRF room to work
		pool_size = max(10, self.k * 3)
		
		"""
		Retrieves the top-k most relevant text chunks for the given sentence.
		"""
		# Query the vector database using k-NN
		# ChromaDB automatically handles the embedding of the query string here
		results = self.collection.query(
			query_texts=[clean_query],
			n_results=pool_size
		)
		# Extract and format the retrieved documents
		dense_docs = results["documents"][0] if results["documents"] else []
		
		# 3. Sparse Retrieval (Exact Lexical Match via BM25)
		tokenized_query = clean_query.split(" ")
		sparse_docs = self.bm25.get_top_n(tokenized_query, self.corpus_docs, n=pool_size)
		
		# 4. Reciprocal Rank Fusion (RRF)
		rrf_scores = {}
		
		# Score Dense Docs
		for rank, doc in enumerate(dense_docs):
			if doc not in rrf_scores: rrf_scores[doc] = 0.0
			rrf_scores[doc] += 1.0 / (60.0 + rank + 1)
			
		# Score Sparse Docs
		for rank, doc in enumerate(sparse_docs):
			if doc not in rrf_scores: rrf_scores[doc] = 0.0
			rrf_scores[doc] += 1.0 / (60.0 + rank + 1)
			
		# 5. Sort by RRF Score
		sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
		top_k_docs = [doc for doc, score in sorted_docs[:self.k]]
		
		# Join the chunks into a single context string
		context_string = " | ".join(top_k_docs)
		
		return context_string

# --- Quick Test Block ---
if __name__ == "__main__":
	retriever = RAGRetriever(k=3)
	
	sample_sentence = "The patient was prescribed Sumatriptan for acute migraines."
	
	print(f"\nOriginal Query: {sample_sentence}")
	print(f"Extracted Keywords: {retriever.extract_keywords(sample_sentence)}\n")
	
	context = retriever.retrieve_context(sample_sentence)
	print(f"Retrieved Hybrid Context:\n{context}")