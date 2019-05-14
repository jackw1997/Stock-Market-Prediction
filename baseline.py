import numpy as np

def baseline():
    data = np.loadtxt("reduced_data.csv", delimiter=",")
    label = np.loadtxt("train_label.csv")
    correct = 0
    for i in range(30000): 
        tn_60 = 0
        tn_10 = 0
        for j in range(60):
            tn_60 += data[i+j][108]
            if j >= 50:
                tn_10 += data[i+j][108]
        tn_60 = tn_60 / 60.0
        tn_10 = tn_10 / 10.0
        result = tn_60 - tn_10
        if i % 1000:
            print(tn_60, tn_10)
        if (result >= 0 and label[i] == 0) or (result < 0 and label[i] == 1):
            correct += 1
    return correct / 30000.0


if __name__ == "__main__":
    print(baseline())