import numpy as np
import Machine_Learning_Algoithm_Base_Sklearn_Dictionary as MLD
from sklearn.neural_network import MLPClassifier,MLPRegressor
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,r2_score

filepath_root="D:\sudty\论文过程数据\图片Sand\\txt数据\\"

def Load_Data(ipath):
	filelist = os.listdir(ipath)  # 获得所有的文件的名字
	filelist = [x for x in filelist if "txt" in x]  # 只要txt的文件
	filename_list = []
	All_data_list=[]
	for x in filelist:
		filename = x.split('.')[0]
		# print(filename,x)
		filename_list.append(filename)
		fpath = os.path.join(ipath, x)
		print(fpath)
		data=np.loadtxt(fpath,delimiter='\t',dtype=str)
		data=data[:,0:-1]
		data=data.ravel()
		All_data_list.append(data)
	sand = pd.read_csv("D:\sudty\Data_MHS\\Y9砂地比.txt", delimiter='\t')
	Well = sand.loc[:, "Well"]
	SDB = sand.loc[:, "SDB"]
	SDB_dic = {}
	for x in range(0, len(Well)):
		SDB_dic[Well.iat[x]] = SDB.iat[x]
	# print("SDBIDC",SDB_dic)
	data_list = []
	SDB_list = []
	n=-1
	for x in filename_list:
		n += 1
		# print("x",x)
		sdb = SDB_dic.get(x, "None")
		if sdb == "None":
			continue
		data_list.append(All_data_list[n])
		SDB_list.append(sdb)
	return data_list,SDB_list

def Run():
	data_attri,data_label=Load_Data(filepath_root)
	data_attri=StandardScaler().fit_transform(data_attri)
	train_attri,test_attri,train_label,test_label=train_test_split(data_attri,data_label,test_size=0.3,random_state=32)
	# MLP=MLPClassifier()
	MLPClassifier()
	# MLP.fit(train_attri,train_label)
	# pred=MLP.predict(test_attri)
	# acc=accuracy_score(test_label,pred)
	# print("ACC:", acc)
	MLPR=MLPRegressor()
	MLPR.fit(train_attri,train_label)
	predR=MLPR.predict(test_attri)
	r2=r2_score(test_label,predR)
	print("R2:",r2)
	# print(data_attri[5][0:20],data_label[5])
if __name__=="__main__":
	Run()