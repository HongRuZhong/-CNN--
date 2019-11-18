import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
filepath="D:\sudty\Data_MHS\Log_Y9_ALL_论证\\"

def Load_Data(ipath):
	filelist = os.listdir(ipath)  # 获得所有的文件的名字
	filelist = [x for x in filelist if "txt" in x]  # 只要txt的文件
	Max_RowNum=0
	data_dic={}
	filename_list=[]
	for x in filelist:
		filename=x.split('.')[0]
		# print(filename,x)
		filename_list.append(filename)
		fpath=os.path.join(ipath,x)
		data=pd.read_csv(fpath,delimiter='\t')
		data=data.loc[:,"AC":"RT"].values
		data=StandardScaler().fit_transform(data)
		if data.shape[0]>Max_RowNum:
			Max_RowNum=data.shape[0]
		data_dic[filename]=data

	#对数据补零
	for x in filename_list:
		data=data_dic[x]
		if data.shape[0]<Max_RowNum:
			pad_m=np.zeros((Max_RowNum-data.shape[0],data.shape[1]))
			data=np.row_stack((data,pad_m))
			data_dic[x]=data
			# print("PAD_0",data.shape)

	sand=pd.read_csv("D:\sudty\Data_MHS\\Y9砂地比.txt",delimiter='\t')
	Well=sand.loc[:,"Well"]
	SDB=sand.loc[:,"label"]-1
	SDB_dic={}
	for x in range(0,len(Well)):
		SDB_dic[Well.iat[x]]=SDB.iat[x]
	# print("SDBIDC",SDB_dic)
	data_list=[]
	SDB_list=[]
	for x in filename_list:
		# print("x",x)
		sdb=SDB_dic.get(x,"None")
		if sdb=="None":
			continue
		data_list.append(data_dic[x].T)
		SDB_list.append(sdb)

	return data_list,SDB_list

class TrainSet(Dataset):
	def __init__(self, data,SDB):
		# 定义好 image 的路径

		self.data, self.label = data, SDB

	def __getitem__(self, index):
		return self.data[index], self.label[index]

	def __len__(self):
		return len(self.data)

def Torch_Dataloader(data,label):
	dataset = TrainSet(data,label)
	trainloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, num_workers=4)
	return trainloader

def Get_Y9_Data():
	data,SDB=Load_Data(filepath)
	train_attri,test_attri,train_label,test_label = train_test_split(data,SDB, test_size=0.3, random_state=1)
	# print(len(train_attri))
	trainloader=Torch_Dataloader(train_attri,train_label)
	testloader=Torch_Dataloader(test_attri,test_label)
	return trainloader,testloader
if __name__ == '__main__':
	trainloader,testloader=Get_Y9_Data()
	for data,index in trainloader:
		print(data[0,:,0:10],index)

