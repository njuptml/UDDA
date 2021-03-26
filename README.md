#UDDA
#### 环境
- Pytorch 1.0
- Python 3.7


#### Dataset

下载GPCRs数据集，百度网盘：https://pan.baidu.com/s/1rlXoDZmHjPryqJ4UjDcQVg，提取码: x3t2 

####特征提取
打开特征提取文件夹，按照里面的readme文件操作

####训练
将上述特征提取中提取好特征的csv文件放在UDDA文件夹下
打开data_loader.py,functions.py,main.py,model.py,test.py，
将main.py中train_data和val_data中的数据集名称改为相应的训练集和测试集数据集名称，
运行main.py文件，就能得到相应的R2和RMSE的值。
