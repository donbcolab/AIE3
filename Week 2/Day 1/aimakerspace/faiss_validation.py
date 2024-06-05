import faiss
import numpy as np

# Step 1: Initialize Parameters
d = 64  # Dimension of each vector
nb = 1000  # Initial number of vectors in the database
nq = 10  # Number of query vectors

# Step 2: Generate Initial Random Data
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')  # Database vectors
xq = np.random.random((nq, d)).astype('float32')  # Query vectors

# Step 3: Build the Initial FAISS Index
index = faiss.IndexFlatL2(d)  # Create an index using L2 (Euclidean) distance
index.add(xb)  # Add the initial database vectors to the index

# Step 4: Perform an Initial Search
k = 4  # Number of nearest neighbors to retrieve
D, I = index.search(xq, k)  # Search the index with the query vectors

# Step 5: Display Search Results
print("Initial Search Results:")
print("Distances:\n", D)  # D contains the distances to the nearest neighbors
print("Indices:\n", I)  # I contains the indices of the nearest neighbors in the database

# Step 6: Define Function to Add New Vectors and Update the Index
def add_to_index(new_vectors: np.array, existing_vectors: np.array, faiss_index: faiss.IndexFlatL2) -> np.array:
    """
    Adds new vectors to the existing vectors and updates the FAISS index.

    Args:
        new_vectors (np.array): New vectors to be added.
        existing_vectors (np.array): Existing database vectors.
        faiss_index (faiss.IndexFlatL2): The FAISS index to be updated.

    Returns:
        np.array: Updated array of database vectors.
    """
    # Add new vectors to the existing database vectors
    updated_vectors = np.vstack((existing_vectors, new_vectors))
    
    # Update the FAISS index with new vectors
    faiss_index.add(new_vectors)
    
    return updated_vectors

# Step 7: Generate New Random Data to be Added
new_vectors = np.random.random((10, d)).astype('float32')  # New vectors to be added

# Step 8: Add New Vectors to the Database and Update the Index
xb = add_to_index(new_vectors, xb, index)

# Step 9: Verify the Index Size After Adding New Vectors
print(f"Index size after adding new vectors: {index.ntotal}")

# Step 10: Perform Another Search to Verify Updates
D_new, I_new = index.search(xq, k)  # Search the updated index with the same query vectors

# Step 11: Display Updated Search Results
print("Updated Search Results:")
print("Distances:\n", D_new)  # D_new contains the distances to the nearest neighbors after the update
print("Indices:\n", I_new)  # I_new contains the indices of the nearest neighbors in the updated database
