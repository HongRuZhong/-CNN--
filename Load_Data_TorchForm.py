import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TrainSet(Dataset):
	def __init__(self, data):
		# 定义好 image 的路径
		self.data, self.label = data[:,0:-1], data[:,-1]

	def __getitem__(self, index):
		return self.data[index], self.label[index]

	def __len__(self):
		return len(self.data)

def Load_Data(ipath):
	data=pd.read_csv(ipath,delimiter='\t')
	data=data.loc[:,"AC":"LIMSTONE"]
	return data.values
	# print(data.head())
	# np.loadtxt(ipath)

def Torch_Dataloader(data):
	dataset = TrainSet(data)
	trainloader = DataLoader(dataset=dataset, batch_size=512, shuffle=True, num_workers=4)
	return trainloader

def Get_Logging_TorchData():
	train_path="D:\Learn_Pytorch\RNN\\train.txt"
	test_path="D:\Learn_Pytorch\RNN\\test.txt"
	train_data=Load_Data(train_path)

	test_data=Load_Data(test_path)

	trainloader=Torch_Dataloader(train_data)
	testloader=Torch_Dataloader(test_data)
	return trainloader,testloader

if __name__=="__main__":
	t1,t2=Get_Logging_TorchData()
	for data,index in t1:
		print(index)
	# main()