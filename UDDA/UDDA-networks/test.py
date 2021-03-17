import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from data_loader import GetLoader
from torchvision import datasets
import sklearn.metrics as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math




def test(val_loader, my_net, epoch):
    # assert dataset_name in ['MNIST', 'mnist_m']

    model_root = 'models'
    # image_root = os.path.join('dataset', dataset_name)

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0

    """load data"""


    """ training """

    # my_net = torch.load(os.path.join(
    #     model_root, 'mnist_mnistm_model_epoch_' + str(epoch) + '.pth'
    # ))
    my_net = my_net.eval()

    # if cuda:
    #     my_net = my_net.cuda()

    len_dataloader = len(val_loader)
    data_target_iter = iter(val_loader)

    i = 0
    n_total = 0
    mae_loss = 0
    mse_loss = 0
    r2_loss = 0
    #loss = []
    #loss_mse = torch.nn.MSELoss()
    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.FloatTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)

        class_output, _ = my_net(input_data=input_img, alpha=alpha)
        #pred = class_output.data.max(1, keepdim=True)[1]
        # n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        n_total += 1
        #loss_test = loss_mse(class_output, class_label)
        #loss.append(loss_test.detach().data.cpu().numpy())
        i += 1
        mae_loss += round(sm.mean_absolute_error(class_label.detach().data.view_as(class_output).cpu(),class_output.detach().cpu()),2)
        mse_loss += round(sm.mean_squared_error(class_label.detach().data.view_as(class_output).cpu(),class_output.detach().cpu()),2)
        #r2_loss += round(sm.r2_score(class_label.detach().data.view_as(class_output).cpu(),class_output.detach().cpu()),2)
        def R2(x, y):
            xmean = torch.mean(x)
            ymean = torch.mean(y)
            s1 = torch.sum((x - xmean) * (y - ymean)) ** 2
            s2 = torch.sum((x - xmean) ** 2) * torch.sum((y - ymean) ** 2)
            return s1 / s2

        def RMSE(x,y):
            return torch.sqrt(torch.mean((x-y)**2))


    mae_loss = mae_loss * 1.0 / n_total
    mse_loss = mse_loss * 1.0 /n_total
    #r2_loss = r2_loss * 1.0/n_total
    rmse = RMSE(class_label.detach().data.view_as(class_output),class_output.detach()) * 1.0 /n_total
    r2_loss = R2(class_label.detach().data.view_as(class_output),class_output.detach()) * 1.0 /n_total

    #print ('epoch %d of mean_absolute_error: %f' % (epoch, accu))
    print ('epoch {} | loss of mae: {} | loss of mse : {} | loss of rmse : {} | loss of r2 : {} '.format(epoch, mae_loss, mse_loss,rmse, r2_loss))
