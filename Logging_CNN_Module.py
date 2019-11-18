
import torch
import torch.nn as nn
import torch.nn.functional as F

# 编写卷积+bn+relu模块
class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channals, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channals, **kwargs)
        self.bn = nn.BatchNorm1d(out_channals)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

# 编写Inception模块
class Inception(nn.Module):
	def Nmm_conv(self,in_planes,out_c,k):
		conv_list=[]
		l=int(k/2.0)
		for x in range(l):
			conv_list.append(BasicConv1d(out_c,out_c,kernel_size=3,padding=1))
		block=nn.Sequential(
			BasicConv1d(in_planes, out_c, kernel_size=1),
			*conv_list
		)
		return block

	def __init__(self, in_planes,
				 n1x1, n3x3,n5x5, n7x7,pool_planes):
		super(Inception, self).__init__()
		# 1x1 conv branch
		self.b1 = BasicConv1d(in_planes, n1x1, kernel_size=1)

		self.b3=self.Nmm_conv(in_planes,n3x3,3)
		self.b5 = self.Nmm_conv(in_planes, n5x5, 5)
		self.b7=self.Nmm_conv(in_planes,n7x7,7)
		# self.b11=self.Nmm_conv(in_planes,n11x11,11)
		self.b_pool = nn.MaxPool2d(3, stride=1, padding=1)
		self.bp = BasicConv1d(in_planes, pool_planes,
								  kernel_size=1)



	def forward(self, x):
		y1 = self.b1(x)
		y2 = self.b3(x)
		y3 = self.b5(x)
		y4 = self.b7(x)
		y5=self.bp(x)
		# y的维度为[batch_size, out_channels, C_out,L_out]
		# 合并不同卷积下的特征图
		return torch.cat([y1, y2, y3, y4,y5], 1)

#编写Resnet模块
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class Net_Res_1D(nn.Module):
	def FC_Layers(self,in_c,out_c):
		block=nn.Sequential(
			nn.Linear(in_c,1024),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(1024,256),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(256,out_c),
			# nn.Sigmoid()
		)
		return block
	def __init__(self,num_classes=10):
		super(Net_Res_1D, self).__init__()
		self.conv1=ResidualBlock(6,64,1)
		self.conv2=ResidualBlock(64,256,1)
		self.conv3 = ResidualBlock(256, 512, 1)
		self.Mpool1=nn.MaxPool1d(2,2)
		self.Mpool2=nn.MaxPool1d(2,2)
		self.Mpool3 = nn.MaxPool1d(2, 2)
		self.lv = 256*7
		self.FC=self.FC_Layers(self.lv,num_classes)

	def forward(self, input):
		# print(input.shape)
		x=self.conv1(input)
		x=self.Mpool1(x)
		x=self.conv2(x)
		x=self.Mpool2(x)
		# x = self.conv3(x)
		# x = self.Mpool3(x)
		# print("xxxx",x.size())
		x=x.view(x.size(0),self.lv)
		# x=x.flatten(1)  #将4为张量后面的3维拉平
		x=self.FC(x)
		# print("x_v",x)
		return x

#论文net
class NET(nn.Module):
	def CONV_Layers(self,in_c,out_c):
		block=nn.Sequential(
			nn.Conv1d(in_c,out_c,kernel_size=2,padding=1),
			nn.ReLU(),
			nn.Conv1d(out_c,out_c,kernel_size=2,padding=1),
			nn.ReLU(),

			nn.MaxPool1d(2,stride=2)
		)
		return block
	def CONV_Layers_NoPad(self,in_c,out_c):
		block=nn.Sequential(
			nn.Conv1d(in_c,out_c,kernel_size=2,),
			nn.ReLU(),
			nn.Conv1d(out_c,out_c,kernel_size=2,),
			nn.ReLU(),

			nn.MaxPool1d(2,stride=2)
		)
		return block
	def FC_Layers(self,in_c):
		block=nn.Sequential(
			nn.Linear(in_c,2500),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(2500,1500),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(1500,6)
		)
		return block
	def __init__(self,num_classes):
		super(NET,self).__init__()
		self.conv1=self.CONV_Layers(1,64)
		self.conv2=self.CONV_Layers_NoPad(64,128)
		self.conv3 = nn.Conv1d(1,64,2,padding=1)
		self.conv4 = nn.Conv1d(64, 64, 2, padding=1)
		self.lv = 128
		self.fc=self.FC_Layers(self.lv)

	def forward(self, input):
		# print(input.shape)
		x=self.conv1(input)
		# print("xx",x.shape)
		x=self.conv2(x)
		# print("xxxx",x.size())
		x=x.view(x.size(0),self.lv)
		# x=x.flatten(1)  #将4为张量后面的3维拉平
		x=self.fc(x)
		return x



class Net_8W_Inception(nn.Module):
	def CONV_Layers(self,in_c,out_c,ker_size=3):
		block=nn.Sequential(
			nn.Conv1d(in_c,out_c,kernel_size=ker_size,),
			nn.BatchNorm1d(out_c),
			nn.ReLU(),
		)
		return block
	def FC_Layers(self,in_c,out_c):
		block=nn.Sequential(
			nn.Linear(in_c,1024),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(1024,256),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(256,out_c),
			# nn.Sigmoid()
		)
		return block
	def __init__(self,num_classes):
		super(Net_8W_Inception,self).__init__()
		incep_k=[4,8,16,8,4]
		incep_channel=0
		for x in incep_k:
			incep_channel+=x
		incep_channel=int(incep_channel)
		self.incep1=Inception(6,incep_k[0],incep_k[1],incep_k[2],incep_k[3],incep_k[4])
		self.incep2=Inception(incep_channel,8,16,32,16,8)
		self.conv1=self.CONV_Layers(incep_channel,128,3)
		self.conv2=self.CONV_Layers(128,256,3)
		self.conv3 = self.CONV_Layers(256,512,3)
		self.Mpool=nn.MaxPool1d(2,2)
		self.Mpoo2 = nn.MaxPool1d(2, 2)
		self.lv = 80*7
		self.fc=self.FC_Layers(self.lv,num_classes)
	def forward(self, input):
		x=self.incep1(input)
		x=self.Mpool(x)
		x=self.incep2(x)
		# x=self.conv1(x)
		# x=self.conv2(x)
		x=self.Mpoo2(x)
		# print("xxxx",x.size())
		x = x.view(x.size(0), self.lv)
		# x=x.flatten(1)  #将4为张量后面的3维拉平
		x = self.fc(x)
		# print("x_v",x)
		return x

class Net_8W_Inception_2(nn.Module):
	def CONV_Layers(self,in_c,out_c,ker_size=3):
		block=nn.Sequential(
			nn.Conv1d(in_c,out_c,kernel_size=ker_size,),
			nn.BatchNorm1d(out_c),
			nn.ReLU(),
		)
		return block
	def FC_Layers(self,in_c,out_c):
		block=nn.Sequential(
			nn.Linear(in_c,1024),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(1024,256),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(256,out_c),
			# nn.Sigmoid()
		)
		return block
	def __init__(self,num_classes):
		super(Net_8W_Inception_2,self).__init__()
		self.incep1 = Inception(6, 8, 16, 32, 4, 4)
		self.incep2=Inception(8+16+32+4+4,16,32,32,8,8)
		self.incep3 = Inception(16 + 32 + 32 + 8 + 8, 32, 64, 64, 16, 16)
		self.Mpool=nn.MaxPool1d(2,2)
		self.Mpoo2 = nn.MaxPool1d(2, 2)
		self.Mpoo3 = nn.MaxPool1d(2, 2)
		self.lv = 192*3
		self.fc=self.FC_Layers(self.lv,num_classes)
	def forward(self, input):
		x=self.incep1(input)
		x=self.Mpool(x)
		x=self.incep2(x)
		x=self.Mpoo2(x)
		x=self.incep3(x)
		x=self.Mpoo3(x)
		# print("xxxx",x.size())
		x = x.view(x.size(0), self.lv)
		# x=x.flatten(1)  #将4为张量后面的3维拉平
		x = self.fc(x)
		# print("x_v",x)
		return x

class NET_MultiChannel(nn.Module):
	def CONV_Layers(self,in_c,out_c,ker_size=3):
		block=nn.Sequential(
			nn.Conv1d(in_c,out_c,kernel_size=ker_size,),
			nn.ReLU(),
			nn.Conv1d(out_c,out_c,kernel_size=ker_size,),
			nn.ReLU(),

			nn.MaxPool1d(2,stride=2)
		)
		return block
	def CONV_Layers_NoPad(self,in_c,out_c,ker_size):
		block = nn.Sequential(
			nn.Conv1d(in_c, out_c, kernel_size=ker_size, ),
			nn.ReLU(),
			# nn.Sigmoid(),
			nn.MaxPool1d(2, stride=2)
		)
		return block

	def Conv2d_Layers(self,in_c,out_c,ker_size):
		block=nn.Sequential(
			nn.Conv2d(in_c,in_c,kernel_size=ker_size,padding=1),
			nn.ReLU(),
			nn.Conv2d(in_c,out_c,kernel_size=ker_size,padding=1),
			nn.ReLU(),
		)
		return block
	def FC_Layers(self,in_c,out_c):
		block=nn.Sequential(
			nn.Linear(in_c,1024),
			# nn.ReLU(),
			# nn.Dropout(0.5),
			nn.Linear(1024,256),
			nn.ReLU(),
			# nn.Dropout(0.5),
			nn.Linear(256,out_c),
			# nn.Sigmoid()
		)
		return block
	def __init__(self,num_classes):
		super(NET_MultiChannel,self).__init__()
		self.conv1=self.CONV_Layers(6,512,7)
		self.conv2=self.CONV_Layers(512,64,3)
		self.conv3 = self.CONV_Layers(64,128,7)
		self.conv21 = self.CONV_Layers_NoPad(6,32,21)
		self.conv22=self.CONV_Layers_NoPad(32,64,13)
		self.conv23=self.CONV_Layers_NoPad(64,128,7)
		self.conv24=self.CONV_Layers_NoPad(128,256,7)
		self.conv25 = self.CONV_Layers_NoPad(256, 512, 7)
		self.conv26 = self.CONV_Layers_NoPad(512, 64,3)
		#2Dconv
		self.conv2d1=self.Conv2d_Layers(1,32,3)
		self.conv2d2=self.Conv2d_Layers(32,128,3)
		self.conv2d3=self.Conv2d_Layers(128,64,3)
		self.mpool=nn.MaxPool2d(2,stride=2)
		self.lv = 64*8
		self.fc=self.FC_Layers(self.lv,num_classes)

	def forward(self, input):
		# print(input.shape)
		x=self.conv21(input)
		x=self.conv22(x)
		x=self.conv23(x)
		x = self.conv24(x)
		x = self.conv25(x)
		x = self.conv26(x)
		# x=self.conv2d1(input)
		# x=self.mpool(x)
		# x=self.conv2d2(x)
		# x=self.conv2d3(x)
		# x=self.mpool(x)
		# x=self.conv23(x)
		# x=self.conv24(x)
		# print("xx",x.shape)
		# x=self.conv2(x)
		# print("xxx",x.shape)
		# x=self.conv3(x)
		# print("xxxx",x.size())
		x=x.view(x.size(0),self.lv)
		# x=x.flatten(1)  #将4为张量后面的3维拉平
		x=self.fc(x)
		# print("x_v",x)
		return x

class NET_8wdata_MultiChannel_1(nn.Module):
	def CONV_Layers(self,in_c,out_c,ker_size=3):
		block=nn.Sequential(
			nn.Conv1d(in_c,out_c,kernel_size=ker_size,),
			nn.BatchNorm1d(out_c),
			nn.ReLU(),
		)
		return block

	def FC_Layers(self,in_c,out_c):
		block=nn.Sequential(
			nn.Linear(in_c, 2500),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(2500, 1500),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(1500, out_c),
			# nn.Sigmoid()
		)
		return block
	def __init__(self,num_classes):
		super(NET_8wdata_MultiChannel_1,self).__init__()
		self.conv1=self.CONV_Layers(6,64,17)
		self.conv2=self.CONV_Layers(64,256,3)
		self.conv3 = self.CONV_Layers(256,512,3)
		self.Mpool=nn.MaxPool1d(2,2)
		self.Mpoo2 = nn.MaxPool1d(2, 2)
		self.lv = 256*2
		self.fc=self.FC_Layers(self.lv,num_classes)

	def forward(self, input):
		# print(input.shape)
		x=self.conv1(input)
		x=self.Mpool(x)
		x=self.conv2(x)
		x=self.Mpoo2(x)
		# print("xxxx",x.size())
		x=x.view(x.size(0),self.lv)
		# x=x.flatten(1)  #将4为张量后面的3维拉平
		x=self.fc(x)
		# print("x_v",x)
		return x

class NET_8wdata_MultiChannel_2(nn.Module):
	def CONV_Layers(self,in_c,out_c,ker_size=3):
		block=nn.Sequential(
			nn.Conv1d(in_c,out_c,kernel_size=ker_size,),
			nn.BatchNorm1d(out_c),
			nn.ReLU(),
		)
		return block

	def FC_Layers(self,in_c,out_c):
		block=nn.Sequential(
			nn.Linear(in_c, 2500),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(2500, 1500),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(1500, out_c),
			# nn.Sigmoid()
		)
		return block
	def __init__(self,num_classes):
		super(NET_8wdata_MultiChannel_2,self).__init__()
		self.conv1=self.CONV_Layers(6,64,3)
		self.conv2=self.CONV_Layers(64,128,3)
		self.conv3 = self.CONV_Layers(128,256,3)
		self.conv4 = self.CONV_Layers(256, 512, 3)
		self.conv5 = self.CONV_Layers(512, 256, 3)
		self.Mpool=nn.MaxPool1d(2,2)
		self.Mpoo2 = nn.MaxPool1d(2, 2)
		self.lv = 256*4
		self.fc=self.FC_Layers(self.lv,num_classes)

	def forward(self, input):
		# print(input.shape)
		x=self.conv1(input)
		x=self.conv2(x)
		x=self.conv3(x)
		x = self.Mpool(x)
		x=self.conv4(x)
		x = self.conv5(x)
		x=self.Mpoo2(x)


		# print("xxxx",x.size())
		x=x.view(x.size(0),self.lv)
		# x=x.flatten(1)  #将4为张量后面的3维拉平
		x=self.fc(x)
		# print("x_v",x)
		return x

class NET_8wdata_MultiChannel_3(nn.Module):
	def CONV_Layers(self,in_c,out_c,ker_size=3):
		block=nn.Sequential(
			nn.Conv1d(in_c,out_c,kernel_size=ker_size,),
			# nn.BatchNorm1d(out_c),
			nn.ReLU(),
		)
		return block

	def FC_Layers(self,in_c,out_c):
		block=nn.Sequential(
			nn.Linear(in_c,1024),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(1024,256),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(256,out_c),
			# nn.Sigmoid()
		)
		return block
	def __init__(self,num_classes,flat_len=None):
		super(NET_8wdata_MultiChannel_3,self).__init__()
		self.conv1=self.CONV_Layers(6,64,7)
		self.conv2=self.CONV_Layers(64,256,3)
		self.Mpool=nn.MaxPool1d(2,2)
		# self.Mpoo2 = nn.MaxPool1d(2, 2)
		self.lv = 256*5
		if flat_len!=None:
			self.lv=256*flat_len
		self.fc=self.FC_Layers(self.lv,num_classes)

	def forward(self, input):
		# print(input.shape)
		x=self.conv1(input)
		x=self.Mpool(x)
		x=self.conv2(x)
		x=self.Mpool(x)
		# print("xxxx,lv",x.size(),self.lv)
		x=x.view(x.size(0),self.lv)
		# x=x.flatten(1)  #将4为张量后面的3维拉平
		x=self.fc(x)
		# print("x_v",x)
		return x

class NET_8wdata_MultiChannel_4(nn.Module):
	def CONV_Layers(self,in_c,out_c,ker_size=3):
		block=nn.Sequential(
			nn.Conv1d(in_c,out_c,kernel_size=ker_size,),
			nn.BatchNorm1d(out_c),
			nn.ReLU(),
		)
		return block

	def FC_Layers(self,in_c,out_c):
		block=nn.Sequential(
			nn.Linear(in_c,1024),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(1024,256),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(256,out_c),
			# nn.Sigmoid()
		)
		return block
	def __init__(self,num_classes):
		super(NET_8wdata_MultiChannel_4,self).__init__()
		self.conv1=self.CONV_Layers(6,64,11)
		self.conv2=self.CONV_Layers(64,128,7)
		self.conv3 = self.CONV_Layers(128, 256, 3)
		self.Mpool=nn.MaxPool1d(2,2)
		self.Mpoo2 = nn.MaxPool1d(2, 2)
		self.lv = 256*2
		self.fc=self.FC_Layers(self.lv,num_classes)

	def forward(self, input):
		# print(input.shape)
		x=self.conv1(input)
		x = self.conv2(x)
		x=self.Mpool(x)
		x=self.conv3(x)
		x=self.Mpoo2(x)
		# print("xxxx",x.size())
		x=x.view(x.size(0),self.lv)
		# x=x.flatten(1)  #将4为张量后面的3维拉平
		x=self.fc(x)
		# print("x_v",x)
		return x

class NET_8wdata_MultiChannel_5(nn.Module):
	def CONV_Layers(self,in_c,out_c,ker_size=3):
		block=nn.Sequential(
			nn.Conv1d(in_c,out_c,kernel_size=ker_size,),
			nn.BatchNorm1d(out_c),
			nn.ReLU(),
		)
		return block

	def FC_Layers(self,in_c,out_c):
		block=nn.Sequential(
			nn.Linear(in_c,256),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(256,64),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(64,out_c),
			# nn.Sigmoid()
		)
		return block
	def __init__(self,num_classes):
		super(NET_8wdata_MultiChannel_5,self).__init__()
		self.conv1=self.CONV_Layers(6,128,7)
		self.conv2=self.CONV_Layers(128,256,5)
		self.conv3 = self.CONV_Layers(256, 128, 3)
		self.Mpool=nn.MaxPool1d(2,2)
		self.Mpoo2 = nn.MaxPool1d(2, 2)
		self.lv = 128*4
		self.fc=self.FC_Layers(self.lv,num_classes)

	def forward(self, input):
		# print(input.shape)
		x=self.conv1(input)
		x = self.conv2(x)
		x=self.Mpool(x)
		x=self.conv3(x)
		x=self.Mpoo2(x)
		# print("xxxx",x.size())
		x=x.view(x.size(0),self.lv)
		# x=x.flatten(1)  #将4为张量后面的3维拉平
		x=self.fc(x)
		# print("x_v",x)
		return x

class NET_8wdata_MultiChannel_7(nn.Module):
	def CONV_Layers(self,in_c,out_c,ker_size=3):
		block=nn.Sequential(
			nn.Conv1d(in_c,out_c,kernel_size=ker_size,),
			nn.BatchNorm1d(out_c),
			nn.ReLU(),
		)
		return block

	def FC_Layers(self,in_c,out_c):
		block=nn.Sequential(
			nn.Linear(in_c,1024),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(1024,128),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(128,out_c),
			# nn.Sigmoid()
		)
		return block
	def __init__(self,num_classes):
		super(NET_8wdata_MultiChannel_7,self).__init__()
		self.conv1=self.CONV_Layers(6,64,2)
		self.conv2=self.CONV_Layers(64,256,2)
		self.conv3 = self.CONV_Layers(256, 512, 2)
		self.conv4 = self.CONV_Layers(512, 64, 2)
		self.Mpool=nn.MaxPool1d(2,2)
		self.Mpoo2 = nn.MaxPool1d(2, 2)
		self.lv = 64*3
		self.fc=self.FC_Layers(self.lv,num_classes)

	def forward(self, input):
		# print(input.shape)
		x=self.conv1(input)

		x=self.conv2(x)
		x = self.Mpool(x)
		x=self.conv3(x)
		x=self.conv4(x)
		x=self.Mpoo2(x)
		# print("xxxx",x.size())
		x=x.view(x.size(0),self.lv)
		# x=x.flatten(1)  #将4为张量后面的3维拉平
		x=self.fc(x)
		# print("x_v",x)
		return x

class NET_8wdata_MultiChannel_8(nn.Module):
	def CONV_Layers(self,in_c,out_c,ker_size=3):
		block=nn.Sequential(
			nn.Conv1d(in_c,out_c,kernel_size=ker_size,),
			nn.BatchNorm1d(out_c),
			nn.ReLU(),
		)
		return block

	def FC_Layers(self,in_c,out_c):
		block=nn.Sequential(
			nn.Linear(in_c, 1024),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(1024, 256),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(256, out_c),
			# nn.Sigmoid()
		)
		return block
	def __init__(self,num_classes):
		super(NET_8wdata_MultiChannel_8,self).__init__()
		self.conv1=self.CONV_Layers(6,64,7)
		self.conv2=self.CONV_Layers(64,256,3)

		self.Mpool=nn.MaxPool1d(2,2)
		self.Mpoo2 = nn.MaxPool1d(2, 2)
		self.lv = 256*5
		self.fc=self.FC_Layers(self.lv,num_classes)

	def forward(self, input):
		# print(input.shape)
		x=self.conv1(input)
		x = self.Mpool(x)
		x=self.conv2(x)
		x = self.Mpoo2(x)

		# print("xxxx",x.size())
		x=x.view(x.size(0),self.lv)
		# x=x.flatten(1)  #将4为张量后面的3维拉平
		x=self.fc(x)
		# print("x_v",x)
		return x

class Net_8W_2DCNN(nn.Module):
	def CONV_Layers(self,in_c,out_c,ker_size=3):
		block=nn.Sequential(
			nn.Conv2d(in_c,out_c,kernel_size=ker_size,),
			nn.BatchNorm2d(out_c),
			nn.ReLU(),
		)
		return block
	def FC_Layers(self,in_c,out_c):
		block=nn.Sequential(
			nn.Linear(in_c,1024),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(1024,256),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(256,out_c),
			# nn.Sigmoid()
		)
		return block
	def __init__(self,num_classes):
		super(Net_8W_2DCNN,self).__init__()
		self.conv1=self.CONV_Layers(1,64,3)
		self.conv2=self.CONV_Layers(64,256,3)
		self.Mpool=nn.MaxPool2d(2,2)
		self.Mpoo2 = nn.MaxPool2d(2, 2)
		self.lv = 256*1*13
		self.fc=self.FC_Layers(self.lv,num_classes)
	def forward(self, input):
		x=self.conv1(input)
		x=self.conv2(x)
		x=self.Mpool(x)
		# print("xxxx",x.size())
		x = x.view(x.size(0), self.lv)
		# x=x.flatten(1)  #将4为张量后面的3维拉平
		x = self.fc(x)
		# print("x_v",x)
		return x