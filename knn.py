import numpy as np
import cvxopt
import matplotlib.pyplot as plt

class KNN(object):
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

    def _split_data(self):
        n_samples, _ = self.sample.shape
        sample_size = int(0.7 * n_samples)
        self.test_sample = self.sample[sample_size:]
        self.sample = self.sample[:sample_size]
        self.test_label = self.label[sample_size:]
        self.label = self.label[:sample_size]

    def dis(self, a, b):
        l = a.shape[0]
        d = 0
        for i in range(l):
            d += (a[i] - b[i]) ** 2
        return d

    def train(self, k = 5):
        self.t = []
        self.d = []
        d = [1e10] * k
        label = [0] * k

        def find_max_index(l):
            m, x = 0, 0
            for i in range(len(l)):
                if l[i] > m:
                    m = l[i]
                    x = i
            return x
        
        correct = 0
        for i in range(self.test_sample.shape[0]):
            for j in range(self.sample.shape[0]):
                x = find_max_index(d)
                m_d = self.dis(self.sample[j], self.test_sample[i])
                if m_d < d[x]:
                    d[x] = m_d 
                    label[x] = self.label[j]
            s = sum(label)
            if (s > k / 2 and self.test_label[i] == 1) or (s <= k / 2 and self.test_label[i] == 0):
                correct += 1
            self.t.append(correct / (i+1) * 1.0)
            self.d.append(i+1)
            if (i % 100 == 0 and i != 0) or i < 20:
                print("Till Interation ", i, ", the correct rate is ", correct / (i+1) * 1.0)
                plt.plot(self.d, self.t)
                plt.xlabel("Iter")
                plt.ylabel("Accuracy")
                plt.title("KNN")
                plt.savefig('knn.png')
        print("The Final Correct Rate is ", correct / self.test_sample.shape[0] * 1.0)

        plt.plot(self.d, self.t)
        plt.xlabel("Iter")
        plt.ylabel("Accuracy")
        plt.title("KNN")
        plt.savefig('knn.png')


if __name__ == "__main__":
    knn = KNN("train_data.csv", "train_label.csv")
    knn.train()
                
        