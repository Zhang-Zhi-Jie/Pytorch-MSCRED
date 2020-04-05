import torch
import numpy as np
import os

splits = ["train", "test"]
train_data_path = "./data/matrix_data/train_data/"
test_data_path = "./data/matrix_data/test_data/"
shuffle = {'train': True, 'test': False}

def load_data():
    dataset = {}
    train_file_list = os.listdir(train_data_path)
    test_file_list = os.listdir(test_data_path)
    train_file_list.sort(key = lambda x:int(x[11:-4]))
    test_file_list.sort(key = lambda x:int(x[10:-4]))
    train_data, test_data = [],[]
    for obj in train_file_list:   
        train_file_path = train_data_path + obj
        train_matrix = np.load(train_file_path)
        #train_matrix = np.transpose(train_matrix, (0, 2, 3, 1))
        train_data.append(train_matrix)

    for obj in test_file_list:
        test_file_path = test_data_path + obj
        test_matrix = np.load(test_file_path)
        #test_matrix = np.transpose(test_matrix, (0, 2, 3, 1))
        test_data.append(test_matrix)

    dataset["train"] = torch.from_numpy(np.array(train_data)).float()
    dataset["test"] = torch.from_numpy(np.array(test_data)).float()

    dataloader = {x: torch.utils.data.DataLoader(
                                dataset=dataset[x], batch_size=1, shuffle=shuffle[x]) 
                                for x in splits}
    return dataloader