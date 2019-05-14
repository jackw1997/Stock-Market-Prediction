import numpy as np
import cvxopt
import matplotlib.pyplot as plt

class Kmeans(object):
    def __init__(self, s_address, l_address):
        self.s_address = s_address
        self.l_address = l_address
        '''
        Here the sample and label means the location of sample and label
        We will use the location to open the dataset
        '''
        self._read_data()
        #self._split_data()

    def _read_data(self):
        self.sample = np.loadtxt(self.s_address, dtype = float, delimiter = ' ')
        self.label = np.loadtxt(self.l_address, dtype = float, delimiter = ' ')
        #self.sample = self.sample[:4123]
        #self.label = self.label[:4123]
        n_samples, n_features = self.sample.shape
        for i in range(n_features):
            if i % 10 == 0:
                print("Started Normalizing Feature: ", i)
            x_min, x_max, x_sum = self.sample[0][i], self.sample[0][i], 0
            for j in range(n_samples):
                x_sum += self.sample[j][i]
                if self.sample[j][i] > x_max:
                    x_max = self.sample[j][i]
                if self.sample[j][i] < x_min:
                    x_min = self.sample[j][i]
            avg = x_sum / n_samples * 1.0
            for j in range(n_samples):
                self.sample[j][i] = (self.sample[j][i] - avg) / (x_max - x_min) 

    def dis(self, a, b):
        l = a.shape[0]
        d = 0
        for i in range(l):
            d += (a[i] - b[i]) ** 2
        return d


    def train(self):
        self.t = []
        self.d = range(100)
        self.k = 2
        self.a = np.random.rand(self.sample.shape[1])
        self.b = np.random.rand(self.sample.shape[1])
        for epoch in range(100):
            cluster_a = []
            label_a = 0
            cluster_b = []
            label_b = 0
            for i in range(self.sample.shape[0]):
                if self.dis(self.a, self.sample[i]) > self.dis(self.b, self.sample[i]):
                    cluster_a.append(self.sample[i])
                    label_a += self.label[i]
                else:
                    cluster_b.append(self.sample[i])
                    label_b += self.label[i]
            self.a = np.zeros(self.sample.shape[1])
            self.b = np.zeros(self.sample.shape[1])
            for s_a in cluster_a:
                for f in range(self.sample.shape[1]):
                    self.a[f] += s_a[f] / len(cluster_a) * 1.0
            for s_b in cluster_b:
                for f in range(self.sample.shape[1]):
                    self.b[f] += s_b[f] / len(cluster_b) * 1.0

            if label_a / len(cluster_a) * 1.0 > label_b / len(cluster_b) * 1.0:
                accuracy = (label_a + len(cluster_b) - label_b) / (len(cluster_a) + len(cluster_b)) * 1.0
            else:
                accuracy = (label_b + len(cluster_a) - label_a) / (len(cluster_a) + len(cluster_b)) * 1.0
            if epoch % 5 == 0:
                print("The accuracy at epoch ", epoch, " is ", accuracy)
            self.t.append(accuracy)

        plt.plot(self.d, self.t)
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.title("K-means")
        plt.show()


if __name__ == "__main__":
    kmeans = Kmeans("train_data.csv", "train_label.csv")
    kmeans.train()
