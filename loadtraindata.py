import cv2
import pandas as pd
import numpy as np




# Function to Load Data

def load_data(filename, data, loop_begin, loop_end):
    for i in range(loop_begin, loop_end):
        path = "images/" + str(filename.iloc[i][0])
        img = cv2.imread(path)
        img = cv2.resize(img, (240, 320))
        data.append(img)
    return data


train = pd.read_csv("train.csv")

train_data = []

batch_num_start = 0
batch_num_end = 772
batch_size = 105

for i in range(batch_num_start, batch_num_end):
    batch_data = []
    train_data += load_data(train, batch_data, batch_size * i, batch_size * (i+1))
    print(f"Batch {i+1}: Data Loaded")

train_data = np.array(train_data)

np.save("trainweights.npy", train_data)
