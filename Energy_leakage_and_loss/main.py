import torch
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from Energy_Reconstruction import *
import random
import os.path
import torch.nn as nn
from training_functions import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

BATCH_SIZE = 64
model_name = "cvt"
device = get_default_device()
print(device)

timestamp = time.strftime("%Y%m%d-%H%M")


data_dir = "../"



xtrain_loader = np.load(data_dir + "electron_x_full.npy").astype('double')
xtrain_loader=torch.tensor(xtrain_loader,dtype=torch.float)
xtrain_loader = (xtrain_loader[:,0,:,:]+xtrain_loader[:,1,:,:])
xtrain_loader,b1=torch.sort(xtrain_loader,2,descending=True)
print(xtrain_loader.shape)

en3dtrain_loader = np.load(data_dir + "electron_en3d_full.npy")
momtrain_loader = np.load(data_dir + "electron_mom_full.npy")
momtrain_loader=torch.from_numpy(momtrain_loader)


train_loader = TensorDataset(xtrain_loader, momtrain_loader)
train_loader = DataLoader(train_loader, batch_size=BATCH_SIZE, shuffle=False)

lr=0.001
epochs=10
run_training=1

n=50000
plt.hist2d(momtrain_loader.detach().cpu().numpy()[:n], en3dtrain_loader[:n], bins=400,cmin=1)
plt.xlabel('Gen Mom Energy(GeV)')
plt.ylabel('Recon Energy(GeV/C)')
plt.title('Momentum vs  en3d Histogram of electron')
cb = plt.colorbar()
cb.set_label('Number of Particles')
plt.show()


model=leak_dist_double_matrix_exp_parameter()


device = get_default_device()  #1

name_trained_model=" "#"checkpoints/cvt_20220501-1514_epoch-1.pt"
if (os.path.isfile(name_trained_model)):
    checkpoint = torch.load(name_trained_model) if torch.cuda.is_available() else torch.load(name_trained_model, map_location=device)
    model.load_state_dict(checkpoint['state_dict']) if os.path.isfile(name_trained_model) else print(" ")
    print("trained model is uploaded",str(name_trained_model))

if torch.cuda.device_count() > 1:
    print("Number of GPUs being used:", torch.cuda.device_count())
    model = nn.DataParallel(model)  #2

model.to(device) #3
optimizer = torch.optim.Adam(model.parameters(), lr)

run_training=1 # if it is 1, it will train and validate
best_loss=10
limit_patience=10
patience=0
if(run_training==1):
    for epoch in range(epochs):
        print(" ")
        print("Training Started")
        model.train()
        training(train_loader,1,model,optimizer)

        print("Epoch", epoch+1, "   completed", (epoch+1) / epochs)
model.eval()

my_output = model(to_device(xtrain_loader[:n],device))
cmap = plt.get_cmap('plasma')
#cmap.set_under('white')

name="Electron"
plt.hist2d(en3dtrain_loader[:n], my_output.detach().cpu().numpy(), bins=400,cmin=1)
plt.xlabel('Reconstructed Energy(GeV)')
plt.ylabel('My Energy(GeV/C)')
plt.title('Reconstructed Energy vs My Energy Histogram of'+name)
cb = plt.colorbar()
cb.set_label('Number of Particles')
plt.show()


plt.hist2d(momtrain_loader.detach().cpu().numpy()[:n], my_output.detach().cpu().numpy(), bins=400,cmin=1)
plt.xlabel('Gen Mom Energy(GeV)')
plt.ylabel('My Energy(GeV/C)')
plt.title('Momentum vs My Energy Histogram of'+name)
cb = plt.colorbar()
cb.set_label('Number of Particles')
plt.show()

