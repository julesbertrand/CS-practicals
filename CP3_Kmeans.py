import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import seaborn as sns


class KMeans:
    def __init__(self, n_clusters, tol=1e-4, max_iter=1000, init_type="random"):
        self.params = {
            "n_clusters": n_clusters,
            "tol": tol,
            "max_iter": max_iter,
            "init_type": init_type,
        }

    @property
    def n_clusters(self):
        return self.params["n_clusters"]

    @n_clusters.setter
    def n_clusters(self, n_clusters):
        if not isinstance(n_clusters, int):
            raise TypeError("Expected: int")
        elif n_clusters <= 0:
            raise TypeError("Expected: int greater that 0")
        self.params["n_clusters"] = n_clusters

    @property
    def tol(self):
        return self.params["tol"]

    @property
    def max_iter(self):
        return self.params["max_iter"]

    @property
    def init_type(self):
        return self.params["init_type"]

    def get_params(self):
        return self.params

    def set_params(self, params):
        if not isinstance(params, dict):
            raise TypeError("Expected: dict")
        if not params:
            return self
        for key, val in params.items():
            if key in self.params.keys():
                self.params[key] = val
            else:
                raise KeyError("This parameter does not exist: {s}".format(key))

    def initialize(self, init_type=None):
        if init_type:
            self.init_type = init_type
        if self.init_type == "random":
            centroids_idx = np.random.choice(
                self.n, size=self.n_clusters, replace=False
            )
            self.centroids = np.array([self.data[c] for c in centroids_idx])
        else:
            raise ValueError("init_type: {} not yet define".format(init_type))

    @staticmethod
    def euclidean_dist(x, y):
        return np.linalg.norm(x - y)

    def fit(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")
        self.data = data
        self.n, self.m = self.data.shape

        # initializing centroids
        self.initialize()

        iter = 0
        diff = float("inf")
        while iter < self.max_iter and diff > self.tol:
            # init clusters
            self.clusters = [[] for i in range(self.n_clusters)]
            # clusters assignments
            for i in range(self.n):
                dists = [self.euclidean_dist(self.data[i], c) for c in self.centroids]
                self.clusters[np.argmin(dists)].append(i)

            # find new cluster centroids
            new_centroids = np.zeros((self.n_clusters, self.m))
            for i, cluster in enumerate(self.clusters):
                new_centroids[i] = np.mean(self.data[cluster], axis=0)

            diff = np.linalg.norm(np.array(self.centroids) - np.array(new_centroids))
            self.centroids = new_centroids
            iter += 1

    def plot_clusters(self):
        colors = sns.color_palette("Set1")
        for i in range(self.n_clusters):
            plt.scatter(
                self.data[self.clusters[i]][:, 0],
                self.data[self.clusters[i]][:, 1],
                color=colors[i],
                s=2,
                label="cluster {}".format(i),
            )
            plt.scatter(
                self.centroids[i, 0],
                self.centroids[i, 1],
                marker="x",
                color=colors[i],
                s=50,
            )
        plt.legend()
        plt.show()

    def predict(self, data):
        if self.centroids is None:
            raise NameError("Please fit model before trying to predict")
        if data.shape[-1] != self.m:
            raise ValueError(
                "Dimension mismatch: centroids and data are not of same dimension"
            )
        centroids = np.zeros(data.shape)
        for i in range(len(data)):
            dists = [self.euclidean_dist(data[i], c) for c in self.centroids]
            centroids[i] = self.centroids[np.argmin(dists)]
        return centroids

    def elbow_method(self, n_clusters_list, data, plot_results=True):
        n_clusters_list = list(n_clusters_list)
        distortions = []
        print(" Starting elbow method ".center(100, "-"))
        for k in n_clusters_list:
            print("fitting for k={}".format(k))
            self.n_clusters = k
            self.fit(data)
            # sum over the min for each data point of distance to centroids
            distortion = sum(np.min(cdist(data, self.centroids, "euclidean"), axis=1))
            distortions.append(distortion)
        if plot_results:
            self.elbow_plot(n_clusters_list, distortions)
        return distortions

    @staticmethod
    def elbow_plot(n_clusters_list, distortions):
        plt.plot(n_clusters_list, distortions, "bx-")
        plt.xlabel("Values of K")
        plt.ylabel("Distortion")
        plt.title("The Elbow Method using Distortion")
        plt.show()


class DataGeneration:
    def __init__(self, size=100):
        self.data = self.ambers_random_data(size=size)

    def ambers_random_data(self, size):
        print(" Generating data ".center(100, "-"))
        x = 2
        data1 = np.random.normal(size=(size, 2)) + [x, x]
        data2 = np.random.normal(size=(size, 2)) + [x, -x]
        data3 = np.random.normal(size=(size, 2)) + [-x, -x]
        data4 = np.random.normal(size=(size, 2)) + [-x, x]
        data = np.concatenate((data1, data2, data3, data4))
        np.random.shuffle(data)
        print(" Data generated ".center(100, "-"))
        return data


if __name__ == "__main__":
    generator = DataGeneration(size=100)
    k_means = KMeans(n_clusters=4)
    k_means.elbow_method(n_clusters_list=range(1, 10), data=generator.data)
    k = input("Choose number of clusters: ")
    k_means.n_clusters = int(k)
    k_means.fit(generator.data)
    k_means.plot_clusters()
    print(k_means.predict(np.array([[0, 1], [1, 0]])))
