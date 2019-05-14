import numpy as np
import pandas as pd

def data_reconstruct(file_path):
    data = np.loadtxt(file_path, delimiter=",")
    print(data.shape)
    my_data = np.zeros((10000, 138*2))
    my_label = np.zeros(10000)
    for i in range(10000):
        for j in range(138):
            my_data[i][j] = data[i+60][j] - data[i][j]
            my_data[i][j+138] = data[i+60][j] - data[i+50][j]
    for i in range(10000):
        midprice_t = data[i+60][108]
        midprice_tn = data[i+70][108]
        if i % 1000 == 0:
            print(midprice_t, midprice_tn)
        if midprice_t >= midprice_tn:
            my_label[i] = 0 #fall
        else:
            my_label[i] = 1 #rise
    np.savetxt("train_data.csv", my_data)
    np.savetxt("train_label.csv", my_label)


if __name__ == "__main__":
    data_reconstruct("reduced_data.csv")