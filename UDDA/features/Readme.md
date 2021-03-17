1.数据都在‘GPCRs’中

2.安装Rdkit用来处理生成ECFP（RDKit_2016_03_1在文件夹里）

Configure "RDKit" environment variables
	you need install python2.7 first("Anaconda" will be convenient) https://www.anaconda.com/download/
	a) put "RDKit_2016_03_1" to disk C
	b) new a system path called "RDBASE" and key is "C:\RDKit_2016_03_1"
	c) new a system path called "PYTHONPATH" and key is "%RDBASE%"
	d) add "%RDBASE%\lib" to path
	you can test your whether your RDKit works by type "from rdkit import Chem" on your python console.


3.处理数据

把分子式单独放在一个excel表中，名字为‘Input_Smiles.xlsx’
把活性值单独放在一个excel表中，名字为‘Response.xlsx’

 

4.生成ECFP

运行‘ECFP_8.py’脚本，输入上一步的’Input_Smiles.xlsx’，输出一个‘xxxx.csv’文件

“xxxx”是自己选择一个名字

5.处理成DNN需要的格式

运行‘DNN_prepare.py’脚本，输入‘xxxx.csv’和‘Response.xlsx’，输出 ‘xxxx_training.csv’和‘xxxx_test.csv’

 


代码中输出文件保存的路径自己选择

