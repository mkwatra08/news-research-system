"""
Vector store implementation for semantic search.
Supports both FAISS (local) and Pinecone (cloud) vector databases.
"""

import os
import pickle
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import faiss
from openai import AsyncOpenAI

from app.utils.config import get_vector_store_config, get_openai_config

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Represents a search result from vector store.
    """
    id: str
    score: float
    metadata: Dict[str, Any]
    content: Optional[str] = None


@dataclass
class Document:
    """
    Represents a document to be stored in vector store.
    """
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class VectorStore(ABC):
    """
    Abstract base class for vector store implementations.
    """
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs that were added
        """
        pass
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        k: int = 10, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    async def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents from vector store.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_document_count(self) -> int:
        """
        Get the total number of documents in the store.
        
        Returns:
            Number of documents
        """
        pass


class EmbeddingService:
    """
    Service for generating embeddings using OpenAI.
    """
    
    def __init__(self):
        """Initialize the embedding service."""
        config = get_openai_config()
        self.client = AsyncOpenAI(api_key=config["api_key"])
        self.model = config["embedding_model"]
        
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        try:
            # Process texts in batches to avoid API limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = await self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    async def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0]


class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store implementation.
    """
    
    def __init__(self, index_path: str, dimension: int = 1536):
        """
        Initialize FAISS vector store.
        
        Args:
            index_path: Path to store FAISS index
            dimension: Embedding dimension
        """
        self.index_path = index_path
        self.dimension = dimension
        self.index = None
        self.metadata_store = {}
        self.id_to_index_mapping = {}
        self.embedding_service = EmbeddingService()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Load existing index or create new one
        self._load_or_create_index()
    
    def _load_or_create_index(self) -> None:
        """Load existing FAISS index or create a new one."""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                
                # Load metadata
                metadata_path = self.index_path + ".metadata"
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        data = pickle.load(f)
                        self.metadata_store = data.get('metadata', {})
                        self.id_to_index_mapping = data.get('id_mapping', {})
                
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                logger.info("Created new FAISS index")
                
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            # Create new index as fallback
            self.index = faiss.IndexFlatIP(self.dimension)
    
    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            metadata_path = self.index_path + ".metadata"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata_store,
                    'id_mapping': self.id_to_index_mapping
                }, f)
            
            logger.debug("FAISS index saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to FAISS vector store."""
        try:
            # Generate embeddings if not provided
            texts_to_embed = []
            for doc in documents:
                if doc.embedding is None:
                    texts_to_embed.append(doc.content)
            
            if texts_to_embed:
                embeddings = await self.embedding_service.generate_embeddings(texts_to_embed)
                embedding_idx = 0
                for doc in documents:
                    if doc.embedding is None:
                        doc.embedding = embeddings[embedding_idx]
                        embedding_idx += 1
            
            # Prepare vectors for FAISS
            vectors = np.array([doc.embedding for doc in documents], dtype=np.float32)
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vectors)
            
            # Add to index
            start_idx = self.index.ntotal
            self.index.add(vectors)
            
            # Store metadata and ID mappings
            added_ids = []
            for i, doc in enumerate(documents):
                faiss_idx = start_idx + i
                self.metadata_store[faiss_idx] = {
                    'id': doc.id,
                    'content': doc.content,
                    **doc.metadata
                }
                self.id_to_index_mapping[doc.id] = faiss_idx
                added_ids.append(doc.id)
            
            # Save index
            self._save_index()
            
            logger.info(f"Added {len(documents)} documents to FAISS index")
            return added_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to FAISS: {e}")
            raise
    
    async def search(
        self, 
        query: str, 
        k: int = 10, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents in FAISS."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_single_embedding(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # Search in FAISS
            search_k = min(k * 2, self.index.ntotal)  # Get more results for filtering
            scores, indices = self.index.search(query_vector, search_k)
            
            # Process results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                metadata = self.metadata_store.get(idx, {})
                
                # Apply filters if provided
                if filter_dict:
                    if not self._matches_filter(metadata, filter_dict):
                        continue
                
                results.append(SearchResult(
                    id=metadata.get('id', str(idx)),
                    score=float(score),
                    metadata=metadata,
                    content=metadata.get('content')
                ))
                
                if len(results) >= k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {e}")
            return []
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True
    
    async def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from FAISS (not directly supported, requires rebuild)."""
        try:
            # FAISS doesn't support direct deletion, so we mark as deleted
            deleted_count = 0
            for doc_id in ids:
                if doc_id in self.id_to_index_mapping:
                    faiss_idx = self.id_to_index_mapping[doc_id]
                    if faiss_idx in self.metadata_store:
                        del self.metadata_store[faiss_idx]
                        del self.id_to_index_mapping[doc_id]
                        deleted_count += 1
            
            if deleted_count > 0:
                self._save_index()
                logger.info(f"Marked {deleted_count} documents as deleted")
            
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Failed to delete documents from FAISS: {e}")
            return False
    
    async def get_document_count(self) -> int:
        """Get the number of documents in FAISS store."""
        return len(self.metadata_store)


class PineconeVectorStore(VectorStore):
    """
    Pinecone-based vector store implementation.
    """
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        """
        Initialize Pinecone vector store.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Pinecone index name
        """
        try:
            import pinecone
            
            pinecone.init(api_key=api_key, environment=environment)
            self.index = pinecone.Index(index_name)
            self.embedding_service = EmbeddingService()
            
            logger.info(f"Connected to Pinecone index: {index_name}")
            
        except ImportError:
            raise ImportError("pinecone-client is required for Pinecone vector store")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to Pinecone vector store."""
        try:
            # Generate embeddings if not provided
            texts_to_embed = []
            for doc in documents:
                if doc.embedding is None:
                    texts_to_embed.append(doc.content)
            
            if texts_to_embed:
                embeddings = await self.embedding_service.generate_embeddings(texts_to_embed)
                embedding_idx = 0
                for doc in documents:
                    if doc.embedding is None:
                        doc.embedding = embeddings[embedding_idx]
                        embedding_idx += 1
            
            # Prepare vectors for Pinecone
            vectors = []
            for doc in documents:
                vectors.append({
                    'id': doc.id,
                    'values': doc.embedding,
                    'metadata': {
                        'content': doc.content,
                        **doc.metadata
                    }
                })
            
            # Upsert to Pinecone
            self.index.upsert(vectors)
            
            logger.info(f"Added {len(documents)} documents to Pinecone")
            return [doc.id for doc in documents]
            
        except Exception as e:
            logger.error(f"Failed to add documents to Pinecone: {e}")
            raise
    
    async def search(
        self, 
        query: str, 
        k: int = 10, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents in Pinecone."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_single_embedding(query)
            
            # Search in Pinecone
            response = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Process results
            results = []
            for match in response['matches']:
                results.append(SearchResult(
                    id=match['id'],
                    score=match['score'],
                    metadata=match.get('metadata', {}),
                    content=match.get('metadata', {}).get('content')
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search Pinecone index: {e}")
            return []
    
    async def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from Pinecone."""
        try:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents from Pinecone: {e}")
            return False
    
    async def get_document_count(self) -> int:
        """Get the number of documents in Pinecone index."""
        try:
            stats = self.index.describe_index_stats()
            return stats['total_vector_count']
        except Exception as e:
            logger.error(f"Failed to get Pinecone document count: {e}")
            return 0


def create_vector_store() -> VectorStore:
    """
    Create and return the appropriate vector store based on configuration.
    
    Returns:
        VectorStore instance
    """
    config = get_vector_store_config()
    
    if config["type"] == "pinecone":
        return PineconeVectorStore(
            api_key=config["api_key"],
            environment=config["environment"],
            index_name=config["index_name"]
        )
    else:
        return FAISSVectorStore(
            index_path=config["index_path"],
            dimension=1536  # OpenAI embedding dimension
        )


# Global vector store instance
vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Get the global vector store instance.
    
    Returns:
        VectorStore instance
    """
    global vector_store
    if vector_store is None:
        vector_store = create_vector_store()
    return vector_store