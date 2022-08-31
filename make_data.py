import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

## make electron data and mom data only

#data = np.load("D:\\ams-02_data\\events_test.npz")

#data1 =pd.DataFrame({"y":data['y'],"mom":data['mom'],"x":data['x'].tolist(),"en3d":data['en3d']})
#data1 =pd.DataFrame({"y":data['y'],"x":data['x'].tolist()})
#data1=data1.query("y==1")
#xx=np.stack(data1['x'])

#np.save("electron_mom_full.npy",xx)
#np.save("electron_mom_full.npy",data1['mom'].values)
#np.save("electron_en3d_full.npy",data1['en3d'].values)
#print(data1['mom'].values)



datax = torch.load("../data/x_train_part1.pt")
datay=torch.load("../data/y_train_part1.pt")
datax=datax.cpu().detach().numpy()
datay=datay.cpu().detach().numpy()
mom=np.load("../data/mom_part1.npy")
en3d=np.load("../data/en3d_part1.npy")
print(len(datax))
print(len(datay))
print(len(mom))
print(len(en3d))

data1 =pd.DataFrame({"y":datay,"x":datax.tolist(),"mom":mom})
data1=data1.query("y==1")
xx=np.stack(data1['x'])

np.save("electron_x_part1.npy",xx)
np.save("electron_y_part1.npy",data1['y'].values)
#np.save("electron_en3d_part1.npy",data1['en3d'].values)
np.save("electron_mom_part1.npy",data1['mom'].values)

print(data1)
print(len(data1['y'].values))
