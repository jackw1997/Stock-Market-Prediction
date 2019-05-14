import numpy as np
import matplotlib.pyplot as plt
import math

class LogisticRegression(object):
    def __init__(self, s_address, l_address):
        self.s_address = s_address
        self.l_address = l_address
        '''
        Here the sample and label means the location of sample and label
        We will use the location to open the dataset
        '''
        self._read_data()
        self._split_data()

    def _read_data(self):
        self.sample = np.loadtxt(self.s_address, dtype = float, delimiter = ' ')
        self.label = np.loadtxt(self.l_address, dtype = float, delimiter = ' ')
        #self.sample = self.sample[:4123]
        #self.label = self.label[:4123]
        n_samples, n_features = self.sample.shape
        for i in range(n_features):
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

    def _split_data(self):
        n_samples, _ = self.sample.shape
        sample_size = int(0.7 * n_samples)
        self.test_sample = self.sample[sample_size:]
        self.sample = self.sample[:sample_size]
        self.test_label = self.label[sample_size:]
        self.label = self.label[:sample_size]

    
    def train(self):
        self.t = []
        def logistic_function(w, x, b):
            return 1 / (1 + math.e **((-1) * (dot(w, x) + b)))
        def dot(w, x):
            s = 0
            for i in range(len(x)):
                s += w[i] * x[i]
            return s
        self.w = np.random.rand(self.sample.shape[1])
        self.b = 1
        self.lr = 1e-4
        for epoch in range(100):
            for i in range(self.sample.shape[0]):
                w_copy = np.copy(self.w)
                for j in range(self.sample.shape[1]):
                    self.w[j] = self.w[j] - self.lr * (logistic_function(w_copy, self.sample[i], self.b) - self.label[i]) * self.sample[i][j] * 1.0
                self.b = self.b - self.lr * (logistic_function(w_copy, self.sample[i], self.b) - self.label[i]) * 1.0
                if i % 10000 == 0:
                    correct = 0
                    for k in range(self.test_sample.shape[0]):
                        predict = 1 if (dot(self.w, self.test_sample[k]) + self.b) > 0.5 else 0
                        if predict == self.test_label[k]:
                            correct += 1.0 / self.test_sample.shape[0]
                    print("At epoch ", epoch, ", and iter at ", i, " the accuracy is ", correct)
                    self.t.append(correct)
                    self.d = range(len(self.t))
                    plt.plot(self.d, self.t)
                    plt.xlabel("Iterations")
                    plt.ylabel("Test Accuracy")
                    plt.title("Logistic Regression")
                    plt.savefig('lr.png')


if __name__ == "__main__":
    lr = LogisticRegression("train_data.csv", "train_label.csv")
    lr.train()
