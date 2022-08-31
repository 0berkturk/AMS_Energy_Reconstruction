import torch
import matplotlib.pyplot as plt
import numpy as np
import os.path
import time
from Energy_Reconstruction import *

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def training(train_loader, epochs, model, optimizer):
    print("training function")
    device = get_default_device()
    for epoch in range(epochs):
        k = 0
        loss_data_sum = 0
        for batch in train_loader:
            k = k + 1
            batch = to_device(batch,device)

            images, labels = batch
            optimizer.zero_grad()
            output = model(images)
            #print(output[:2],labels[:2])
            A = torch.nn.L1Loss()
            loss = A(output.float(),labels.float())

            loss.backward()
            optimizer.step()
            loss_data_sum = loss_data_sum + loss
            optimizer.zero_grad()

            if(k % 500==0):
                print(loss_data_sum / k)

        loss_mean = loss_data_sum / k
        print("Loss of Training Data = ", loss_mean)
        print(" ")
        print(" ")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

        print(" ")
        print(" ")
        print(torch.abs(output-labels).detach().cpu().numpy())
        # save loss

        c = (loss_mean).detach().cpu().numpy()
        if os.path.isfile("training_loss.npy"):
            y=np.load("training_loss.npy")
        else:
            y=[]
        np.save("training_loss.npy", np.append(y, c))