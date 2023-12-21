import faiss
import numpy as np


class VecDB:
    def __init__(self, file_path=None, create_index=False):
        self.index = None

        if file_path:
            if create_index:
                # Replace 128 with the dimension of your vectors
                self.index = faiss.IndexFlatL2(70)
            else:
                self.index = faiss.read_index(file_path)

    def add(self, raw_data, num_clusters):
        if self.index is not None:
            return
        # Perform clustering using IVFPQ
        # Assuming raw_data is a 2D array where each row is a vector
        d = 70

        data = list()
        for i in range(len(raw_data)):
            data.append(raw_data[i].get("embed"))

        data = np.array(data).astype('float32')

        quantizer = faiss.IndexFlatL2(d)
        # Adjust parameters as needed
        index = faiss.IndexIVFPQ(quantizer, d, num_clusters, 35, 8)

        # Train the index with the coarse quantizer
        rng = np.random.default_rng(100)
        Xt = rng.random((2500, 70), dtype=np.float32)
        index.train(Xt)

        index.add(data)

        # Save the index to the class member
        self.index = index

        return index

    def search(self, query, top_k, metric='l2'):
        # Perform search using the provided query
        if self.index is None:
            raise ValueError(
                "Index not created. Use add() method to create the index.")

        # Assuming query is a 2D array where each row is a vector
        if metric == 'l2':
            distances, indices = self.index.search(query, top_k)
        elif metric == 'cosine':
            # For cosine similarity, you may need to normalize the vectors
            faiss.normalize_L2(query)
            distances, indices = self.index.search(query, top_k)
        else:
            raise ValueError("Invalid distance metric. Use 'l2' or 'cosine'.")

        return indices[0]
