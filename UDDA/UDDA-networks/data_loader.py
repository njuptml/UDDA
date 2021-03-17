import torch.utils.data as data
from PIL import Image
import os
# import openpyxl as pxl
from torch import nn
import torch.utils.data as data
import csv 
import torch as t
from torch.utils.data import DataLoader

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data


class Protein_set(data.Dataset):
    def __init__(self, root, partition='train'):
        super(Protein_set, self).__init__()
        self.partition = partition
        csv_file = csv.reader(open(root, 'r'))
        self.all_data = []
        for i, line in enumerate(csv_file):
            if i == 0:
                continue
            self.all_data.append([[float(j) for j in line[2:]], float(line[1])])
        self.all_len = len(self.all_data)
        self.data = self.all_data[:int(1.0*self.all_len)] if self.partition == 'train' else self.all_data[int(1.0*self.all_len):]
        
        
    def __getitem__(self, index):
        return t.tensor(self.data[index][0]), t.tensor(self.data[index][1])

    def __len__(self):
        return len(self.data)
            



if __name__ == '__main__':
    # test
    #Data = Protein_set('ECFP5120_training.csv')
    print(len(Data))
    data_loader = DataLoader(Data, batch_size, num_workers=4)
    for i, (data, label) in enumerate(data_loader):
        print(data.size())
        exit()
    
