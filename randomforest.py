from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

class RandomForest(object):

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


    def train(self):
        clf = RandomForestClassifier()
        clf.fit(self.sample, self.label)
        y_ = clf.predict(self.test_sample)
        correct = 0
        for i in range(self.test_sample.shape[0]):
            if y_[i] == self.test_label[i]:
                correct += 1
        print("The accuracy of Random Forest is ", correct / (self.test_sample.shape[0]) * 1.0)
        plt.plot(range(len(clf.feature_importances_)), clf.feature_importances_)
        plt.xlabel("Feature Number")
        plt.ylabel("Importance")
        plt.title("Random Forest")
        plt.show()


if __name__ == "__main__":
    RF = RandomForest("train_data.csv", "train_label.csv")
    RF.train()