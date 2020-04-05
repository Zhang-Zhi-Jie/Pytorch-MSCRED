import torch
import torch.nn as nn
import torch.functional as F 

from model.mscred import MSCRED
from utils.data import load_data


def train(dataLoader, model, criterion, optimizer, epochs):
    print("------training-------")
    for epoch in range(epochs):
        train_l_sum = 0
        for x in dataLoader:
            x = x.squeeze()
            l = criterion(model(x), x[-1].unsqueeze(0)).sum()
            train_l_sum += l
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
        print("[Epoch %d/%d] [loss: %f]" % (epoch+1, epochs, train_l_sum))



if __name__ == '__main__':
    dataLoader = load_data()
    criterion = torch.nn.MSELoss()
    mscred = MSCRED(3, 256)
    optimizer = torch.optim.Adam(mscred.parameters(), lr = 0.0002)
    train(dataLoader["train"], mscred, criterion, optimizer, 10)
