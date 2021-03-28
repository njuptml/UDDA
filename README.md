# UDDA
#### 环境
- Pytorch 1.0
- Python 3.7


#### Dataset

下载GPCRs数据集，百度网盘：https://pan.baidu.com/s/1rlXoDZmHjPryqJ4UjDcQVg，提取码: x3t2 

#### 特征提取
打开features文件夹
1.数据都在‘GPCRs’中
2.安装Rdkit用来处理生成ECFP（RDKit_2016_03_1在文件夹里）
Configure "RDKit" environment variables you need install python2.7 first("Anaconda" will be convenient) https://www.anaconda.com/download/ a) put "RDKit_2016_03_1" to disk C b) new a system path called "RDBASE" and key is "C:\RDKit_2016_03_1" c) new a system path called "PYTHONPATH" and key is "%RDBASE%" d) add "%RDBASE%\lib" to path you can test your whether your RDKit works by type "from rdkit import Chem" on your python console.
3.处理数据
把分子式单独放在一个excel表中，名字为‘Input_Smiles.xlsx’ 把活性值单独放在一个excel表中，名字为‘Response.xlsx’
4.生成ECFP
运行‘ECFP_8.py’脚本，输入上一步的’Input_Smiles.xlsx’，输出一个‘xxxx.csv’文件
![b233e7eece6499814943c83ad4ef115](https://user-images.githubusercontent.com/20634431/112744150-097ba600-8fd0-11eb-9aa3-42e159e5a515.png)
“xxxx”是自己选择一个名字,如下图中命名为“B1_ECFP8_1024.csv”
![7cf91f5e160324ffe5fb71f0462ff6f](https://user-images.githubusercontent.com/20634431/112744159-22845700-8fd0-11eb-8b27-e15c43fdfeb0.png)

5.处理成DNN需要的格式
运行‘DNN_prepare.py’脚本，输入‘xxxx.csv’和‘Response.xlsx’，输出 ‘xxxx_training.csv’和‘xxxx_test.csv’
![0bec90ef2a747a3429afcc1cbd87904](https://user-images.githubusercontent.com/20634431/112744383-38931700-8fd2-11eb-9907-fe7d14018adb.png)
代码中输出文件保存的路径自己选择

#### 训练
1.将上述特征提取中提取好特征的csv文件放在UDDA文件夹下
2.打开data_loader.py,functions.py,main.py,model.py,test.py文件
3.将main.py中train_data和val_data中的数据集名称改为相应的训练集和测试集数据集名称
![ab4cd25e96a3bfde469e1ec1b0f9dd9](https://user-images.githubusercontent.com/20634431/112744392-52345e80-8fd2-11eb-8642-a6edea1d851f.png)
4.运行main.py文件，就能得到相应的R2和RMSE的值。
![3ad82e7e50e930cb70d87a5f0a3ecb6](https://user-images.githubusercontent.com/20634431/112744604-23b78300-8fd4-11eb-918a-90dc3dc4a163.png)

