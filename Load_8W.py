'''导入8w数据，以torch形式，列上取窗口，首先分开井，然后每个点的上下都取S大小的窗口，
组成[batch,n_feature,S]的torch形式'''

import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import My_PythonLib as MP

filepath="D:\Data\Data_机器学习常用数据集\\8W89口井岩性数据\\Data_去除678类.txt"

class DataToTorch(Dataset):
	def __init__(self, data,label):
		# 定义好 image 的路径

		self.data, self.label = data, label

	def __getitem__(self, index):
		return self.data[index], self.label[index]

	def __len__(self):
		return len(self.data)

def Load_Data(ipath):
	data=np.loadtxt(ipath,skiprows=1,dtype=str)
	attri=data[:,3:-1].astype(float)
	attri=StandardScaler().fit_transform(attri)
	data[:,3:-1]=attri.astype(str)
	return data

def Split_Well(data):
	'''分割开每口井'''
	well_list=[]
	sample_list=[]
	well_name = data[0,1]
	# print(well_name)
	for x in data:
		if x[1]!=well_name:
			well_name=x[1]
			wellAll=np.row_stack(sample_list)
			# print(wellAll[0],wellAll[-1])
			well_list.append(wellAll)
			sample_list.clear()
		else:
			sample_list.append(x)
	return well_list

def Get_window_data(well_list,window_size):
	'''得到每个窗口的数据'''
	data_list=[]
	for x in well_list:
		for y in range(window_size,len(x)-window_size):
			w_data=x[y-window_size:y+window_size+1]
			data_list.append(w_data)
		# print(data_list[-1].shape,data_list[-1])
	return data_list

def Del_Well_name_Label(data,window_size):
	'''删除前面没用的列，给每个样本一个标签'''
	attri_list=[]
	label_list=[]
	for x in data:
		attri_list.append(x[:,3:-1].T.astype(float)) #加上转置，转化为（6,31）
		label_list.append(float(x[:,-1][window_size]))
	return attri_list,label_list

def Test_Data(window_size):
	'''測試數據'''
	data = Load_Data(filepath)
	# print(data[0:5])
	well_list = Split_Well(data)
	data_list = []
	for x in well_list:
		for y in range(window_size, len(x) - window_size):
			w_data=x[y,3:]
			data_list.append(w_data)
	data=np.row_stack(data_list)
	train_attri, test_attri, train_label, test_label = train_test_split(data[:,0:-1], data[:,-1], test_size=0.3, random_state=32)
	MP.ML_Model_Run(train_attri,train_label,test_attri,test_label)


def Get_Data_Logging_1DCNN(window_size):
	'''返回最终的数据'''
	data = Load_Data(filepath)
	# print(data[0:5])
	well_list = Split_Well(data)
	data_list=Get_window_data(well_list, window_size)
	attri,label=Del_Well_name_Label(data_list,window_size)
	train_attri,test_attri,train_label,test_label=train_test_split(attri,label,test_size=0.3,random_state=32)
	print(type(train_attri[0]))
	#torch format
	train_set=DataToTorch(train_attri,train_label)
	test_set=DataToTorch(test_attri,test_label)
	train_loader=DataLoader(train_set,batch_size=64,shuffle=True,num_workers=1)
	test_loader=DataLoader(test_set,batch_size=64,shuffle=False,num_workers=1)
	return train_loader,test_loader

def Get_Data_Logging_2DCNN(window_size):
	'''返回最终的数据'''
	data = Load_Data(filepath)
	# print(data[0:5])
	well_list = Split_Well(data)
	data_list=Get_window_data(well_list, window_size)
	attri,label=Del_Well_name_Label(data_list,window_size)
	attri=np.expand_dims(attri,1)
	train_attri,test_attri,train_label,test_label=train_test_split(attri,label,test_size=0.3,random_state=32)
	print(type(train_attri[0]))
	#torch format
	train_set=DataToTorch(train_attri,train_label)
	test_set=DataToTorch(test_attri,test_label)
	train_loader=DataLoader(train_set,batch_size=64,shuffle=True,num_workers=4)
	test_loader=DataLoader(test_set,batch_size=64,shuffle=False,num_workers=4)
	return train_loader,test_loader

def Get_onewell_Data(window_size):
	'''返回某口井的数据进行预测'''
	filepath="D:\投的文章\Paper_基于1DCNN的岩相分类\图鉴\井的岩相\\57-04-0-5.txt"
	data = np.loadtxt(filepath, skiprows=1, dtype=str)
	attri = data[:, 1:-1].astype(float)
	attri = StandardScaler().fit_transform(attri)
	data[:, 1:-1] = attri.astype(str)
	'''得到每个窗口的数据'''
	data_list=[]
	depth_list=[]
	for y in range(window_size,len(data)-window_size):
		w_data=data[y-window_size:y+window_size+1]
		attri=w_data[:,1:-1].T.astype(float)
		label=w_data[window_size,-1].astype(float)
		depth=w_data[window_size,0]
		data_list.append(attri)
		depth_list.append(depth)
	attri=np.array(data_list)
	attri=torch.tensor(attri)
	return attri,np.row_stack(depth_list)
	# print(data_list[-1].shape,data_list[-1])

if __name__=="__main__":
	Get_onewell_Data(15)
	exit()
	windows_size=15
	Test_Data(windows_size)
	exit()
	t1,t2=Get_Data_Logging_1DCNN(windows_size)
	for x,y in t1:
		print(x.shape)