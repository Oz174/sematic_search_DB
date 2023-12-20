import faiss
import numpy as np


class VecDB:
    def __init__(self, file_path=None, new_db=False):
        if file_path:
            self.index = faiss.read_index(file_path)
        else:
            self.index = None

    def insert_records(self, records: list(dict)):
        data = list()
        for i in range(len(records)):
            print("alo")
            data.append(records[i].get("embed"))

        data = np.array(data).astype('float32')
        np.random.seed(42)
        Xt = np.random.rand(2500, 70).astype('float32')
        d = 70  # Dimension of the vectors
        nlist = 100  # Number of clusters
        quantizer = faiss.IndexFlatL2(d)  # Flat index for clustering

        # Create IndexIVFFlat
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

        self.index.train(Xt)
        self.index.add(data)

    def retrieve(self, query, top_k):
        dists, ids = self.index.search(query, top_k)
        return ids[0]
