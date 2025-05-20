from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class SearchResult:
    """Data class to store search results"""
    filename: str
    content: str
    similarity_score: float

class SemanticSearchEngine:
    """
    A semantic search engine that uses sentence transformers for multilingual document search.
    Uses 'paraphrase-multilingual-mpnet-base-v2' by default, which supports 50+ languages.
    """

    def __init__(
        self,
        documents_dir: str,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        device: str = "cpu"
    ):
        self.documents_dir = Path(documents_dir)
        self.model_name = model_name
        self.device = device
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model and initialize storage
        self.model = None
        self.index = None
        self.documents: Dict[int, Tuple[str, str]] = {}  # id -> (filename, content)
        
        self._initialize_search_engine()

    def _initialize_search_engine(self) -> None:
        """Initialize the model and load documents"""
        self.logger.info(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self._load_and_index_documents()

    def _load_and_index_documents(self) -> None:
        """Load all text documents and create FAISS index"""
        # Load all text files
        text_files = list(self.documents_dir.glob("*.txt"))
        if not text_files:
            raise ValueError(f"No text files found in {self.documents_dir}")

        self.logger.info(f"Loading {len(text_files)} documents...")
        documents = []
        
        # Load documents with progress bar
        for idx, file_path in enumerate(tqdm(text_files, desc="Loading documents")):
            try:
                content = file_path.read_text(encoding='utf-8')
                documents.append(content)
                self.documents[idx] = (file_path.name, content)
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {str(e)}")

        # Create embeddings
        self.logger.info("Creating document embeddings...")
        embeddings = self.model.encode(documents, show_progress_bar=True)
        
        # Initialize FAISS index
        vector_dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(vector_dimension)
        self.index.add(embeddings.astype('float32'))
        
        self.logger.info("Search engine initialization complete")

    def search(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.6
    ) -> List[SearchResult]:
        """
        Search for documents most similar to the query.
        
        Args:
            query: Search query (can be in any supported language)
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1) to include in results
            
        Returns:
            List of SearchResult objects sorted by similarity
        """
        # Create query embedding
        query_embedding = self.model.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        # Process results
        results = []
        for idx, (distance, doc_idx) in enumerate(zip(distances[0], indices[0])):
            if doc_idx != -1:  # Valid index
                filename, content = self.documents[doc_idx]
                
                # Convert distance to similarity score (0-1)
                similarity = 1 - (distance / 2)  # Normalize distance to similarity
                
                if similarity >= score_threshold:
                    results.append(SearchResult(
                        filename=filename,
                        content=content,
                        similarity_score=similarity
                    ))
        
        return results


def main():
    """Example usage of the SemanticSearchEngine"""
    # Initialize search engine
    search_engine = SemanticSearchEngine(
        documents_dir="./data",
        model_name="paraphrase-multilingual-mpnet-base-v2"
    )
    
    # Example queries in different languages
    queries = [
        "artificial intelligence applications",  # English
        "inteligencia artificial aplicaciones",  # Spanish
        "künstliche Intelligenz Anwendungen",   # German
    ]
    
    # Perform searches
    for query in queries:
        print(f"\nSearching for: {query}")
        results = search_engine.search(query, top_k=2)
        
        for result in results:
            print(f"\nDocument: {result.filename}")
            print(f"Similarity Score: {result.similarity_score:.3f}")
            print(f"Preview: {result.content[:200]}...")


if __name__ == "__main__":
    # Example usage
    try:
        main()
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")