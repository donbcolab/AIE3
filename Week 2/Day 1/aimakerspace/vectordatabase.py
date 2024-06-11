import faiss
import numpy as np
from typing import List, Tuple, Callable, Union, Optional
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio
import os
import openai

def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """
    Computes the cosine similarity between two vectors.

    Args:
        vector_a (np.array): The first vector.
        vector_b (np.array): The second vector.

    Returns:
        float: The cosine similarity between vector_a and vector_b.
    """
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

class VectorDatabase:
    """
    A class used to represent a Vector Database using FAISS for efficient approximate nearest neighbors search.

    Attributes:
        vectors (np.array): Array of vectors.
        metadata (List[dict]): List of metadata dictionaries.
        embedding_model (EmbeddingModel): Model to generate embeddings.
        index (faiss.IndexFlatL2): FAISS index for efficient vector search.
    """

    def __init__(self, embedding_model: EmbeddingModel = None):
        """
        Initializes the VectorDatabase with an optional embedding model.

        Args:
            embedding_model (EmbeddingModel, optional): The model to generate embeddings. Defaults to None.
        """
        self.vectors = np.empty((0, 512))  # Initialize with an empty array with the appropriate dimension (512 is an example)
        self.metadata = []
        self.embedding_model = embedding_model or EmbeddingModel()
        self.index = None

    def insert(self, key: str, vector: np.array) -> None:
        if self.vectors.size == 0:
            self.vectors = np.array([vector])
        else:
            self.vectors = np.vstack((self.vectors, vector))
        self.metadata.append({'text': key})
        if self.index is not None:
            self.index.add(np.array([vector]))

    def build_index(self) -> None:
        """
        Builds the FAISS index with the existing vectors.
        """
        if self.vectors.size == 0:
            raise ValueError("No vectors to build the index with.")
        embedding_dim = self.vectors.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(self.vectors)

    def search_by_text(self, query: str, k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Searches the vector database by a query text and returns the top k closest matches.

        Args:
            query (str): The query text to search for.
            k (int, optional): The number of top matches to return. Defaults to 5.

        Returns:
            List[Tuple[str, float, dict]]: A list of tuples containing the text, distance, and metadata of the top k closest matches.
        """
        if not self.index:
            raise ValueError("FAISS index has not been built. Please build the index before searching.")

        query_vector = np.array([self.embedding_model.get_embedding(query)])
        distances, indices = self.index.search(query_vector, k)
        results = [(self.metadata[i]['text'], distances[0][idx], self.metadata[i]) for idx, i in enumerate(indices[0])]
        
        return results

    def search_by_text_cosine(self, query: str, k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Searches the vector database by a query text and returns the top k closest matches using cosine similarity.

        Args:
            query (str): The query text to search for.
            k (int, optional): The number of top matches to return. Defaults to 5.

        Returns:
            List[Tuple[str, float, dict]]: A list of tuples containing the text, similarity score, and metadata of the top k closest matches.
        """
        query_vector = np.array([self.embedding_model.get_embedding(query)])
        similarities = [cosine_similarity(query_vector[0], vec) for vec in self.vectors]
        sorted_indices = np.argsort(similarities)[::-1][:k]
        results = [(self.metadata[i]['text'], similarities[i], self.metadata[i]) for i in sorted_indices]
        
        return results

    def add_vectors(self, new_vectors: np.array, new_metadata: List[dict]) -> None:
        """
        Adds new vectors to the existing vectors and updates the FAISS index.

        Args:
            new_vectors (np.array): New vectors to be added.
            new_metadata (List[dict]): Corresponding metadata for the new vectors.
        """
        if new_vectors.shape[1] != self.vectors.shape[1]:
            raise ValueError(f"New vectors must have the same dimension as existing vectors: {self.vectors.shape[1]}")
        
        self.vectors = np.vstack((self.vectors, new_vectors))
        for metadata in new_metadata:
            self.metadata.append(metadata)
        if self.index is not None:
            self.index.add(new_vectors)

    def get_metadata(self, key: Union[int, str]) -> dict:
        """
        Retrieves metadata for a given key (index or text).

        Args:
            key (Union[int, str]): The key to search for metadata.

        Returns:
            dict: Metadata corresponding to the key.
        """
        if isinstance(key, int):
            return self.metadata[key]
        elif isinstance(key, str):
            for meta in self.metadata:
                if meta['text'] == key:
                    return meta
            raise KeyError(f"No metadata found for text: {key}")
        else:
            raise TypeError("Key must be an integer index or a string text.")

    async def abuild_from_list(self, list_of_text: List[str], metadata_list: Optional[List[dict]] = None) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        if metadata_list is None:
            for text, embedding in zip(list_of_text, embeddings):
                self.insert(text, np.array(embedding))
        else:
            self.vectors = np.array(embeddings)
            for text, metadata in zip(list_of_text, metadata_list):
                metadata['text'] = text  # Ensure each metadata dictionary contains the text
                self.metadata.append(metadata)

        # Build FAISS index
        self.build_index()
        return self

async def main():
    embedding_model = EmbeddingModel()  # Initialize the embedding model

    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    metadata_list = [
        {"title": "Text 1", "source": "Source 1", "date": "2024-01-01"},
        {"title": "Text 2", "source": "Source 2", "date": "2024-02-01"},
        {"title": "Text 3", "source": "Source 3", "date": "2024-03-01"},
        {"title": "Text 4", "source": "Source 4", "date": "2024-04-01"},
        {"title": "Text 5", "source": "Source 5", "date": "2024-05-01"},
    ]

    vector_db = VectorDatabase(embedding_model=embedding_model)
    vector_db = await vector_db.abuild_from_list(list_of_text, metadata_list) #optional metadata_list parameter
    vector_db = await vector_db.abuild_from_list(list_of_text) 

    k = 2

    # Search using FAISS
    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s) using FAISS:", searched_vector)

    # Search using Cosine Similarity
    searched_vector_cosine = vector_db.search_by_text_cosine("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s) using Cosine Similarity:", searched_vector_cosine)

    retrieved_metadata = vector_db.get_metadata("I like to eat broccoli and bananas.")
    print("Retrieved metadata:", retrieved_metadata)

    # Adding new vectors to the database
    new_texts = [
        "Eating healthy is important.",
        "I enjoy hiking in the mountains.",
    ]
    new_metadata = [
        {"title": "Text 6", "source": "Source 6", "date": "2024-06-01", "text": new_texts[0]},
        {"title": "Text 7", "source": "Source 7", "date": "2024-07-01", "text": new_texts[1]},
    ]

    # Generate embeddings for new_texts to use as new_vectors.
    new_vectors = np.array(await embedding_model.async_get_embeddings(new_texts))
    vector_db.add_vectors(new_vectors, new_metadata)

    print(f"Index size after adding new vectors: {vector_db.index.ntotal}")

    # Search again using FAISS
    relevant_texts = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} text(s) using FAISS:", relevant_texts)

    # Search again using Cosine Similarity
    relevant_texts_cosine = vector_db.search_by_text_cosine("I think fruit is awesome!", k=k)
    print(f"Closest {k} text(s) using Cosine Similarity:", relevant_texts_cosine)

if __name__ == "__main__":
    asyncio.run(main())
