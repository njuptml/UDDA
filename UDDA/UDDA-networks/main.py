import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from data_loader import GetLoader, Protein_set
from torchvision import datasets
from torchvision import transforms
from model import CNNModel
import numpy as np
from test import test
from torch.utils.data import DataLoader


model_root = 'models'
cuda = True
cudnn.benchmark = True
lr = 1e-4
batch_size = 128
n_epoch = 2000

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load data

train_data = Protein_set('LS4_ECFP8_1024.csv')
val_data = Protein_set('LT1_ECFP8_1024.csv')
# print(trian_data[0][1].type())
# exit()
dataloader_source = DataLoader(train_data, batch_size, True)
dataloader_target = DataLoader(val_data, batch_size, True)


# load model

my_net = CNNModel()

# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=lr)

# loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()
loss_class = torch.nn.MSELoss()
# loss_domain = torch.nn.MSELoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# training

for epoch in range(n_epoch):

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    while i < len_dataloader:

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label = data_source

        my_net.zero_grad()
        batch_size = len(s_label)

        # input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        input_img = torch.FloatTensor(batch_size, 1024)
        class_label = torch.FloatTensor(batch_size)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()
        # print(s_label.type())
        # exit()
        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)

        class_output, domain_output = my_net(input_data=input_img, alpha=alpha)
        #mse_correct = round(sm.mean_squared_error(class_label.detach().data.cpu(),class_output.detach().cpu()),2)
   
        #print(class_label[:20])
        #print(class_output[:20])
        #print(class_output.size(), class_label.size())
        #exit()
        err_s_label = torch.sqrt(loss_class(class_output.squeeze(), class_label))
        err_s_domain = loss_domain(domain_output, domain_label)
        #print(err_s_label, mse_correct)
        #exit()
        # training model using target data
        data_target = data_target_iter.next()
        t_img, _ = data_target

        batch_size = len(t_img)

        input_img = torch.FloatTensor(batch_size, 1024)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()

        if cuda:
            t_img = t_img.cuda()
            input_img = input_img.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)

        _, domain_output = my_net(input_data=input_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()

        i += 1

        print ('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch, i, len_dataloader, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))

    test(dataloader_target, my_net, epoch)
    test(dataloader_target, my_net, epoch)
    my_net.train()

print('done')
