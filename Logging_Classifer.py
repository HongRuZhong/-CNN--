import torch
import torch.nn as nn
import Logging_CNN.Load_Data_TorchForm as LD
import Logging_CNN.Load_Logging_Y9 as LY9
import Logging_CNN.Load_8W as L8w
import Logging_CNN.Logging_CNN_Module as NET
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import My_PythonLib as MP
import numpy as np
import sklearn.metrics as metrics
from collections import Iterable

import os
import warnings
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter
import PyTorch_Tool_Py.classification_train as train_tool
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

MP.setup_seed(1)

epochs=401

def Run():
	trainloader,testloader=LY9.Get_Y9_Data()
	net=NET.NET_MultiChannel(5)
	criterion = nn.CrossEntropyLoss()
	optimizer_ft = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	# decay LR by a factor of 0.1 every 7 epochs
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1000, gamma=0.1)
	for epoch in range(epochs):
		# for image,label in train_loader:
		# print(image)
		train_tool.train_one_epoch(net, criterion, optimizer_ft, trainloader, device, epoch, print_freq=50)
		exp_lr_scheduler.step()
		if epoch%100==0:
			train_tool.evaluate(net, criterion, testloader, device)
	torch.save(net, "D:\\Net.pkl")
	torch.save(net.state_dict(),"D:\\Net_param.pkl")

Feature_Map_List=[]
def Get_Featuer_Map_of_Model():
	def hook_function(module,grad_in,grad_out):
		Feature_Map_List.append(grad_out)
	opath="D:\\sudty\\1DCNN用于岩性分类\\tensorboard文件\\8wdata_2层1DCNN_73核"
	trainloader, testloader = L8w.Get_Data_Logging_1DCNN(15)
	dataiter = iter(testloader)
	logging, label = dataiter.next()

	net = NET.NET_8wdata_MultiChannel_3(6)
	net.load_state_dict(torch.load(opath+"\\大核model_param.pkl"))

	for x in net.children():
		if isinstance(x,torch.nn.Sequential):
			for y in x:
				if isinstance(y,torch.nn.Conv1d):#判断子层是卷积层还是池化层，fc层，拥这种方式
					y.register_forward_hook(hook_function)
	logging=logging.float()
	out=net(logging)
	orig_data=logging.detach()[0]
	orig_data=F.pad(orig_data,(1,1,1,1),value=1.70141e38).numpy().T
	MP.Sufer_OutGrd(opath+"\\原始特征图.grd",orig_data,-0.5,orig_data.shape[1]-1.5,-0.5,orig_data.shape[0]-1.5)
	print(Feature_Map_List)
	for index,fm in enumerate(Feature_Map_List):
		print(index)
		data=fm.detach()[0]
		data=F.pad(data,(1,1,1,1),value=1.70141e38).numpy().T
		print(data.shape)
		MP.Sufer_OutGrd(opath+"\\第"+str(index+1)+"个卷积层特征图.grd",data,-0.5,data.shape[1]-1.5,
						-0.5,data.shape[0]-1.5)
		print(data.shape)




def Run_8wdata():
	opath="D:\\ZHR_USE\\1DCNN\Results\\8wdata_1DCNN_73核3_Adam_9windowsize"
	writer=SummaryWriter(opath)
	window_size=9
	trainloader,testloader=L8w.Get_Data_Logging_1DCNN(window_size)
	flat_len = int(((window_size * 2 + 1 - 7) / 2 - 2) / 2)
	net = NET.NET_8wdata_MultiChannel_3(6,flat_len)
	#tensorboard
	dataiter = iter(trainloader)
	logging,label=dataiter.next()
	print(logging.shape)
	writer.add_graph(net,(logging.float(),))
	criterion = nn.CrossEntropyLoss()
	optimizer_ft = optim.Adam(net.parameters(), lr=0.001)
	# decay LR by a factor of 0.1 every 7 epochs
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)

	train_loss=[]
	train_predict_label=0
	train_true_label=0
	test_loss=[]
	test_predict_label=0
	test_true_label=0
	test_iter_num=0
	ACC=0  #初始精度
	#输出每次迭代的精度
	fptrainACC=open(os.path.join(opath, "train_ACC.txt"), 'w')
	fptrainACC = open(os.path.join(opath, "train_ACC.txt"), 'a')
	fptestACC = open(os.path.join(opath, "test_ACC.txt"), 'w')
	fptestACC = open(os.path.join(opath, "test_ACC.txt"), 'a')
	for epoch in range(epochs):
		# for image,label in train_loader:
		# print(image)
		train_loss_one_epoch,train_predict_label,train_true_label=train_tool.train_one_epoch(
			net, criterion, optimizer_ft, trainloader, device, epoch, print_freq=200,writer=writer)
		train_loss.append(train_loss_one_epoch)
		# exp_lr_scheduler.step()
		train_ACC = metrics.accuracy_score(train_true_label, train_predict_label)
		print(train_ACC, file=fptrainACC)
		if epoch % 10 == 0:
			test_loss_one_epoch, test_predict_label, test_true_label = train_tool.evaluate(net, criterion, testloader,
																						   device, test_iter_num,writer)
			ACC_one_epoch = metrics.accuracy_score(test_true_label, test_predict_label)
			print(ACC_one_epoch, file=fptestACC)
			if ACC_one_epoch > ACC:
				Out_Result(opath, net, train_predict_label, train_true_label, test_predict_label, test_true_label,epoch)
				ACC = ACC_one_epoch
			test_iter_num += 1
			test_loss.append(test_loss_one_epoch)
	# print(train_loss)
	train_loss = np.array(train_loss)
	np.savetxt(os.path.join(opath, "train_loss.csv"), train_loss, delimiter=',', fmt="%.04f")
	test_loss = np.array(test_loss)
	np.savetxt(os.path.join(opath, "test_loss.csv"), test_loss, delimiter=',', fmt="%.04f")
	writer.close()

def Out_Result(opath, net, train_predict_label, train_true_label, test_predict_label, test_true_label,iter_No=None):
	# 输出最终的评价

	trainACC, train_kappa, train_report, train_cofM = MP.Result_Evaluate(train_predict_label, train_true_label)
	fp = open(os.path.join(opath, "train_report.txt"), 'w')
	print(("train_ACC:" + str(trainACC) + "\n"), file=fp)
	print(("train_kappa:" + str(train_kappa) + "\n"), file=fp)
	print(train_report, file=fp)
	fp.close()
	train_cofM.to_csv(opath + "\\train混淆矩阵.csv")
	testACC, test_kappa, test_report, test_cofM = MP.Result_Evaluate(test_predict_label, test_true_label)
	fp = open(os.path.join(opath, "test_report.txt"), 'w')
	if iter_No!=None:
		print(("第几次迭代：",iter_No),file=fp)
	print(("test_ACC:" + str(testACC) + "\n"), file=fp)
	print(("test_kappa:" + str(test_kappa) + "\n"), file=fp)
	print(test_report, file=fp)
	test_cofM.to_csv(opath + "\\test混淆矩阵.csv")
	fp.close()

	# Save model
	torch.save(net, opath + "\\model整体.pkl")
	torch.save(net.state_dict(), opath + "\\model_param.pkl")

# print(testACC, test_kappa, '\t', test_cofM)
# print(testACC)

def Evaluate_OneWell():
	'''训练好的模型对一口井进行预测，用于出图'''
	opath = "D:\\sudty\\1DCNN用于岩性分类\\tensorboard文件\\8wdata_1DCNN_73核3_Adam"
	net = NET.NET_8wdata_MultiChannel_3(6)
	net.load_state_dict(torch.load(opath+"\\model_param.pkl"))
	net.eval()
	data,depth=L8w.Get_onewell_Data(15)
	data=data.float()
	output=net(data)
	pred = torch.max(output, 1)[1].numpy()
	pred=pred.astype(float)
	depth=depth.astype(float)
	pred=np.column_stack((depth,pred))
	np.savetxt("D:\投的文章\Paper_基于1DCNN的岩相分类\Results\\Adam预测结果.csv",pred)

def Run_Y9():
	trainloader, testloader = LY9.Get_Y9_Data()
	net = NET.NET_MultiChannel(5)
	# criterion = nn.MSELoss(reduction='sum')
	criterion=nn.CrossEntropyLoss()
	LR=0.001
	optimizer_ft = optim.SGD(net.parameters(), lr=LR,)
	# decay LR by a factor of 0.1 every 7 epochs
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1000, gamma=0.1)
	for epoch in range(epochs):
		for data,label in trainloader:
			# print("label2", label)
			label=label.float()
			# print("label",label)
			# data=torch.unsqueeze(data,dim=1)
			# print(data.shape)
			data=data.float()
			data,label=data.to(device),label.to(device)
			output=net(data)
			print("ol",output,label)
			loss=criterion(output,label)
			optimizer_ft.zero_grad()
			loss.backward()
			optimizer_ft.step()
			# R2=metrics.r2_score(output.detach().numpy(),label.detach().numpy())
			print("train:  LR:{}\tepoch:{}\tloss:{}\tR2:{}".format(LR,epoch,loss,R2))
			# exit()
		exp_lr_scheduler.step()

		continue
		if epoch%100==0:
			for data,label in testloader:
				label = label.float()
				data = data.float()
				data, label = data.to(device), label.to(device)
				output = net(data)
				loss = criterion(output, label)
				R2 = metrics.r2_score(output.detach().numpy(), label.detach().numpy())
				print("test:  LR:{}\tepoch:{}\tloss:{}\tR2:{}".format(LR,epoch, loss, R2))

def Run_8wdata_Diff_WindowSize():
	'''此算法可一次导出所有窗口的所有结果'''
	opath="D:\\ZHR_USE\\1DCNN\\Results\\"
	ACC_list=[]
	for i in range(7,10,2):
		subpath=os.path.join(opath,"SGD_"+str(i)+"windowsize_epochs400")
		if not os.path.exists(subpath):
			os.mkdir(subpath)
		trainloader, testloader = L8w.Get_Data_Logging_1DCNN(i)
		flat_len=int(((i*2+1-7)/2-2)/2)
		net = NET.NET_8wdata_MultiChannel_3(6,flat_len)
		net.to(device)
		# tensorboard

		criterion = nn.CrossEntropyLoss()
		optimizer_ft = optim.SGD(net.parameters(), lr=0.001)
		# decay LR by a factor of 0.1 every 7 epochs
		exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
		train_ACC,test_ACC=0,0
		test_iter_num=0
		ACC=0
		train_loss = []
		test_loss = []
		# 输出每次迭代的精度
		fptrainACC = open(os.path.join(subpath, "train_ACC.txt"), 'w')
		fptrainACC = open(os.path.join(subpath, "train_ACC.txt"), 'a')
		fptestACC = open(os.path.join(subpath, "test_ACC.txt"), 'w')
		fptestACC = open(os.path.join(subpath, "test_ACC.txt"), 'a')
		for epoch in range(2):
			# for image,label in train_loader:
			# print(image)
			train_loss_one_epoch, train_predict_label, train_true_label = train_tool.train_one_epoch(
				net, criterion, optimizer_ft, trainloader, device, epoch, print_freq=200)
			train_loss.append(train_loss_one_epoch)
			train_ACC = metrics.accuracy_score(train_true_label, train_predict_label)
			print(train_ACC,file=fptrainACC)
			test_loss_one_epoch, test_predict_label, test_true_label = train_tool.evaluate(net, criterion,
																						   testloader,device, test_iter_num)
			test_loss.append(test_loss_one_epoch)
			ACC_one_epoch = metrics.accuracy_score(test_true_label, test_predict_label)
			print(ACC_one_epoch, file=fptestACC)
			if ACC_one_epoch > ACC:
				Out_Result(subpath, net, train_predict_label, train_true_label, test_predict_label, test_true_label,epoch)
				ACC = ACC_one_epoch

				# train_ACC=metrics.accuracy_score(train_true_label, train_predict_label)
			# test_ACC = metrics.accuracy_score(test_true_label, test_predict_label)
		# print(loss)
		train_loss = np.array(train_loss)
		np.savetxt(os.path.join(subpath, "train_loss.csv"), train_loss, delimiter=',', fmt="%.04f")
		test_loss = np.array(test_loss)
		np.savetxt(os.path.join(subpath, "test_loss.csv"), test_loss, delimiter=',', fmt="%.04f")

		ACC=np.array((i,train_ACC,test_ACC))
		ACC_list.append(ACC)
	odata=np.row_stack(ACC_list)
	np.savetxt(opath+"SGD400迭代次数不同窗口的ACC.txt",odata,fmt="%.04f")

		# print(train_loss)



if __name__=="__main__":
	Run_8wdata_Diff_WindowSize()
	exit()
	# Evaluate_OneWell()
	# exit()
	# Run()
	# exit()
	# Run_Y9()
	Run_8wdata()
	# Get_Featuer_Map_of_Model()
	exit()
	net=torch.load("D:\\Net.pkl")
	net_p=NET.NET_MultiChannel(5)
	net_p.load_state_dict(torch.load("D:\\Net_Param.pkl"))
	n=0
	del net_p.fc
	# print(net_p)
	for k in net_p.children():
		print(k)
	exit()

	for x  in net:
		print(n,x)
		n+=1