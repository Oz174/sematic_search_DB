# from typing import Dict,
import numpy as np
import os

# from scipy.cluster.vq import kmeans2
from sklearn.cluster import MiniBatchKMeans as KMeans


class VecDB:
    def __init__(self, file_path="saved_db", new_db=True) -> None:
        if not os.path.exists(f"./{file_path}"):
            os.mkdir(f"./{file_path}")
        self.file_path = file_path
                

    def string_rep(self, vec):
        return ",".join([str(e) for e in vec])

    def str_rep2_vec(self, vec):
        return "".join(str(int(e * 10)) for e in vec) #create unique ids for each vector stored in a centroid
    
    def save_clusters(self, rows, labels, centroids):
        files = [open(f"./{self.file_path}/cluster_{i}", "a") for i in range(len(centroids))]
        centroid_file_path = f"./{self.file_path}/centroids"
        print('before writing')
        for i in range(len(rows)):
            _id = self.mp[self.str_rep2_vec(rows[i])]
            files[labels[i]].write(f"{_id},{self.string_rep(rows[i])}\n")
        [f.close() for f in files]
        with open(centroid_file_path, "a") as fout:
            for centroid in centroids:
                fout.write(f"{centroid}\n")

    def num_clusters(self, rows_count):
        # walahi kan fe paper ll eqn di bs I lost it , hehe 
        self.no_clusters = int(np.ceil(rows_count / np.sqrt(rows_count)) * 3)
        return self.no_clusters

    def cluster_data(self, rows):
        if type(rows[0]) == dict:
            self.mp = {self.str_rep2_vec(row["embed"]): row["id"] for row in rows}
            print(self.num_clusters(len(rows)))
            rows = [row["embed"] for row in rows]
        else:
            print('Creating mappings from vector to string IDs')
            self.mp = {self.str_rep2_vec(row): i for i, row in enumerate(rows)}

        print('begin clustering')
        kmeans = KMeans(
            n_clusters=self.num_clusters(len(rows)), n_init=1, verbose=True,
            batch_size=int(1e5)
        ).fit(rows)
        print('after clustering')
        labels = kmeans.predict(rows)
        centroids = list(map(self.string_rep, kmeans.cluster_centers_))
        print('before saving clusters')
        self.save_clusters(rows, labels, centroids)
            

    def insert_records(self, rows):
        self.cluster_data(rows)

    def retrive(self, query, top_k=5):
        clusters = []
        data = []
        with open(f"./{self.file_path}/centroids", "r") as fin:
            clusters.extend(
                np.array(list(map(float, line.split(",")))) for line in fin.readlines()
            )
            scores = sorted(
                [
                    (self._cal_score(query, clusters[i])[0], i)
                    for i in range(len(clusters))
                ],
                reverse=True,
            )
            # the numscores serves as _nprobe , which is the number of voroni diagram we use while seacrching for the query
            numscores = 125 if len(clusters) > 3000 else 95 
            top_m_clusters = [open(f"./{self.file_path}/cluster_{i}", "r") for _, i in scores[:numscores]]
            data = []
            for f in top_m_clusters:
                data.extend(
                    [
                        (self._cal_score(query, np.array(list(map(float, line.split(",")[1:])))), int(line.split(",")[0]))
                        for line in f.readlines()
                    ]
                )
            # print(len(data))
            data = sorted(data, reverse=True)
            return [d[1] for d in data[:top_k]]
                
    #cosine_distance
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)
