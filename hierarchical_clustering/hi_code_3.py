import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets.samples_generator import make_blobs
style.use('ggplot')
import numpy as np

centers = [[1,7],
           [5,-2],
           [-10,-10]]
x, _ = make_blobs(n_samples=100,centers = centers, cluster_std=1)
# print(x)

# plt.scatter(x[:,0], x[:,1])
# plt.show()

colors = 10*["g","r","c","b","k","m"]

class Mean_shift:
    def __init__(self, radius= 4):
        self.radius = radius

    def fit(self,data):
        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for feature in data:
                    if np.linalg.norm(feature-centroid) < self.radius:
                        in_bandwidth.append(feature)

                new_centroid = np.average(in_bandwidth,axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))
            prev_centroid = dict(centroids)

            centroids = {}

            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i],prev_centroid[i]):
                    optimized = False

                if not optimized:
                    break
            if optimized:
                break

        self.centroids = centroids

    def predict(self,data):
        pass

clf = Mean_shift()
clf.fit(x)

centroids = clf.centroids
print(centroids)

plt.scatter(x[:,0],x[:,1],s=10)

for c in centroids:
    plt.scatter(centroids[c][0],centroids[c][1],color='k',marker='*',s=50)

plt.show()
