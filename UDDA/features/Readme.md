1.���ݶ��ڡ�GPCRs����

2.��װRdkit������������ECFP��RDKit_2016_03_1���ļ����

Configure "RDKit" environment variables
	you need install python2.7 first("Anaconda" will be convenient) https://www.anaconda.com/download/
	a) put "RDKit_2016_03_1" to disk C
	b) new a system path called "RDBASE" and key is "C:\RDKit_2016_03_1"
	c) new a system path called "PYTHONPATH" and key is "%RDBASE%"
	d) add "%RDBASE%\lib" to path
	you can test your whether your RDKit works by type "from rdkit import Chem" on your python console.


3.��������

�ѷ���ʽ��������һ��excel���У�����Ϊ��Input_Smiles.xlsx��
�ѻ���ֵ��������һ��excel���У�����Ϊ��Response.xlsx��

 

4.����ECFP

���С�ECFP_8.py���ű���������һ���ġ�Input_Smiles.xlsx�������һ����xxxx.csv���ļ�

��xxxx�����Լ�ѡ��һ������

5.�����DNN��Ҫ�ĸ�ʽ

���С�DNN_prepare.py���ű������롮xxxx.csv���͡�Response.xlsx������� ��xxxx_training.csv���͡�xxxx_test.csv��

 


����������ļ������·���Լ�ѡ��

