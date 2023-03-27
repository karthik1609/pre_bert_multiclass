from sklearn import cluster
import numpy as np
import pickle
import time

a = time.time()

arr = np.load('parser_files/arr.npy')

#arr = arr.astype('float16')
print(arr.dtype)
clusterer = cluster.AgglomerativeClustering(
    n_clusters = None,
    #n_clusters=1000, 
    #affinity='cosine', 
    memory=None, 
    connectivity=None, 
    compute_full_tree='auto', 
    linkage='complete', 
    distance_threshold=0.5, 
    compute_distances=False
)
clusterer.fit(arr[np.random.choice(arr.shape[0], 75000, replace=False), :])
print(100. * clusterer.labels_[clusterer.labels_ != -1].shape[0] / clusterer.labels_.shape[0], np.unique(clusterer.labels_).shape[0])

pickle.dump(clusterer, open('cluster_ft_05_euclidean.pkl', 'wb'))

print(time.time() - a)