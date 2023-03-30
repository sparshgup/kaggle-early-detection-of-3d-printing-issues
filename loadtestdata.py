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


test = pd.read_csv("test.csv")

test_data = []

batch_num = 1487
batch_size = int(len(test)/batch_num)

batch_num_start = 0
batch_num_end = 1487

for i in range(batch_num_start, batch_num_end):
    batch_data = []
    test_data += load_data(test, batch_data, batch_size * i, batch_size * (i+1))
    print(f"Batch {i+1}: Data Loaded")

test_data = np.array(test_data)

np.save("testweights.npy", test_data)
