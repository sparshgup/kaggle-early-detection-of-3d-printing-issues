import pandas as pd
import numpy as np



train = pd.read_csv("train.csv")

#Loading Labels
y = []

for i in range(0, len(train)):
    y.append(train.iloc[i][3])


np.savetxt("labels.csv", y, delimiter=",", header="labels")
    
