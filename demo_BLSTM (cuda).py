# try to build a lstm network
import scipy.io as sio
import random
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from torch.nn import functional as F
#import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')

torch.manual_seed(1)
writer = SummaryWriter('run/BLSTM_129-129(L)-256-256-129(L)')

class LSTMNet(nn.Module):

    def __init__(self, featDim, hidden_dim,batch_size):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.linear1 = nn.Linear(featDim,hidden_dim[0])
        self.lstm = nn.LSTM(hidden_dim[0],hidden_dim[1],2,batch_first=True,bidirectional=True)
        #因为是双层RNN，所以output(seq_len, batch, hidden_size * num_directions) 最后的输出的维度是hidden_size*num_directions
        self.linear2 = nn.Linear(hidden_dim[2]*2,hidden_dim[3])

    def init_hidden(self):
        self.h_lstm = (Variable(torch.zeros(4, self.batch_size, self.hidden_dim[1]).cuda()),
              Variable(torch.zeros(4, self.batch_size, self.hidden_dim[1]).cuda())) #4是怎么来的，(num_layers * num_directions, batch, hidden_size)
        return self.h_lstm

    def forward(self, input,sequence):
        layer1 = self.linear1(input)
        layer1 = F.elu(layer1,alpha=1.0, inplace=False)
        packed_layer1 = nn.utils.rnn.pack_padded_sequence(layer1, sequence, batch_first=True)
        layer2, _ = self.lstm(packed_layer1,self.h_lstm)
        unpack_layer3 = nn.utils.rnn.pad_packed_sequence(layer2,batch_first=True)
        unpack_layer3 = unpack_layer3[0]
        layer4 = self.linear2(unpack_layer3)
        return layer4

def shuffle_data(AC,BC,trainNum):
    index = list(range(trainNum))
    random.shuffle(index)
   # print(index)
    AC = AC[index]
    BC = BC[index]
    return AC,BC

#所以prepare_data的作用就是将帧数对齐，返回8*最大帧数*129
def prepare_data(bc,ac,batchsize,featDim):
    DATA = np.zeros((batchsize,2000,featDim))
    LABEL = np.zeros((batchsize,2000,featDim))
    Masking = np.zeros((batchsize,2000,featDim))
    TrueSequence = []
    maxSequence = 1
    for i in range(ac.shape[0]):   #i表示语料条数，所以ac是三维数组
            LABEL[i,:ac[i].shape[0],:] = ac[i]
            DATA[i,:bc[i].shape[0],:] = bc[i]
            Masking[i,:bc[i].shape[0],:] = np.ones(bc[i].shape) #masking全是1，对应骨导的type
            TrueSequence.append(ac[i].shape[0]) #表示每条语音数据的帧的行数
            if ac[i].shape[0]>maxSequence:
                maxSequence = ac[i].shape[0]
    paixu = np.argsort(TrueSequence) #对TrueSequence中的数从小到大排序，返回对应索引
    paixu = paixu[::-1] #将数组反向，也就是上边的序列变成从大到小了
    DATA = DATA[paixu,:maxSequence,:]
    LABEL = LABEL[paixu,:maxSequence,:]
    Masking = Masking[paixu,:maxSequence,:]
    Sequence = np.array(TrueSequence)[paixu] #对TrueSequence进行从大到小排序
    return DATA,LABEL,Sequence,Masking

def log_and_normalize(data,mean,std):
    log_norm_data = []
    #print('data shape[1]:',data.shape[1])
    for i in range(data.shape[1]):
        temp = np.log(data[0][i])
        #print("temp.shape:",temp.shape)
        #print(mean.shape)
        #print(std.shape)
        temp = (temp-mean)/std
        #plt.imshow(temp.T,origin='lower')
        log_norm_data.append(temp)
    return np.array(log_norm_data)


def de_log_and_normalize(data,mean,std):
    temp = data
    temp = temp*std+mean
    temp = np.exp(temp)
    return np.array(temp)

def my_lstm_mse_loss(output,target,sequence,Masking):
    error = output1-LABEL
    error = error*Masking
    error = torch.sum(error**2)/sum(Sequence)
    return error


# load data and split to train and val dataset
TRAIN = sio.loadmat('data/f001_STFT_TRAINSET')
AC = TRAIN['STFT_ac'] # change stft to log
BC = TRAIN['STFT_bc']

dataInfo = sio.loadmat('data/f001_datainfo.mat')
AC_mean,AC_std = dataInfo['log_STFT_ac_mean'],dataInfo['log_STFT_ac_var']
BC_mean,BC_std = dataInfo['log_STFT_bc_mean'],dataInfo['log_STFT_bc_var']
# normalize data
AC = log_and_normalize(AC,AC_mean,AC_std)
BC = log_and_normalize(BC,BC_mean,BC_std)



testdata = sio.loadmat('data/f001_STFT_TESTSET')
t_ac,t_bc = np.array(testdata['STFT_ac']),np.array(testdata['STFT_bc'])
t_ac = log_and_normalize(t_ac,AC_mean,AC_std)
t_bc = log_and_normalize(t_bc,BC_mean,BC_std)



Num = AC.shape[0]
train_ac,train_bc = AC[:],BC[:]
val_ac,val_bc = t_ac,t_bc
train_num = train_ac.shape[0]
val_num = val_ac.shape[0]

train_batchsize = 8
val_batchsize = 8
num_epochs = 50
featDim = 129
hidden_dim = [129,256,256,129] # feedforward hidden, lstm hidden1, lstm hidden2, feedforward hidden
num_train_batch = int(train_num/train_batchsize)
num_val_batch = int(val_num/val_batchsize)

LSTMModel = LSTMNet(featDim, hidden_dim,train_batchsize)
# initial weight
for name, param in LSTMModel.named_parameters():
  if 'bias' in name:
     nn.init.constant_(param, 0.0)
  elif 'weight' in name:
     nn.init.xavier_normal_(param)


# check weight
#LSTMModel.modules().next().LSTM.data
#LSTMModel.lstm1.weight_hh_l0



LSTMModel.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(LSTMModel.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#optimizer = optim.RMSprop(LSTMModel.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=2,factor=0.5,min_lr=0.000001)
num_iteration_train = 0
num_iteration_test = 0
best_model_wts = copy.deepcopy(LSTMModel.state_dict())
best_loss = 1000




notimproveNum = 0
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)
    if notimproveNum>5:
        print('Valloss do not improve at {} epochs,so break'.format(notimproveNum))
        break
    for phase in ['train', 'val']:
        if phase == 'train':
            #scheduler.step()
            LSTMModel.train()  # Set model to training mode
            LSTMModel.batch_size = train_batchsize
            num_batch = num_train_batch
            batchsize = train_batchsize
            AC,BC = shuffle_data(train_ac,train_bc,train_num)
        else:
            LSTMModel.eval() # Set model to evaluate mode
            LSTMModel.batch_size = val_batchsize
            num_batch = num_val_batch
            batchsize = val_batchsize
            AC,BC = shuffle_data(val_ac,val_bc,val_num)
        running_loss = 0.0
        for j in range(num_batch):
            DATA,LABEL,Sequence,Masking = prepare_data(BC[j*batchsize:(j+1)*batchsize],AC[j*batchsize:(j+1)*batchsize],batchsize,featDim) #prepare_data(bc,ac,batchsize,featDim):
            DATA,LABEL,Masking = torch.from_numpy(DATA).float(),torch.from_numpy(LABEL).float(),torch.from_numpy(Masking).float() # Pa
            DATA,LABEL,Masking= Variable(DATA.cuda()),Variable(LABEL.cuda()),Variable(Masking.cuda())
            LSTMModel.zero_grad()
            LSTMModel.hidden = LSTMModel.init_hidden()
            print(DATA.shape)
            #print(Sequence.shape)
            input()
            output1 = LSTMModel(DATA,Sequence)
            loss = criterion(output1, LABEL)
            #loss = my_lstm_mse_loss(output1,LABEL,Sequence,Masking)
            if phase == 'train':
                loss.backward()
                optimizer.step()
                num_iteration_train = num_iteration_train+1
                writer.add_scalar('TrainLoss', loss.item(), num_iteration_train)
            else:
                num_iteration_test = num_iteration_test+1
                writer.add_scalar('VALLoss', loss.item(), num_iteration_test)
            running_loss += loss.item()
            batch_average_loss = running_loss/(j+1)

        epoch_loss = running_loss/(num_batch)
        if phase == 'val':
            former_lr = optimizer.param_groups[0]['lr']
            scheduler.step(epoch_loss)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Epoch_VALLoss', epoch_loss, epoch)
            print('learning rate is {}'.format(optimizer.param_groups[0]['lr']))
            if  epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(LSTMModel.state_dict())
                LSTMModel.load_state_dict(best_model_wts)
                print('BestLoss: {:.4f} is Epoch{} '.format(best_loss,epoch+1))
                notimproveNum = 0
            else:
                notimproveNum = notimproveNum +1
        else:
            writer.add_scalar('Epoch_TrainLoss', epoch_loss, epoch)
        print('{} EpochLoss: {:.4f} '.format(phase,epoch_loss))

LSTMModel.load_state_dict(best_model_wts)
torch.save(LSTMModel.state_dict(),'params.pkl')



# ##------------evaluate------
# load data
TRAIN1 = sio.loadmat('data/f001_STFT_TESTSET')
AC,BC = TRAIN1['STFT_ac'],TRAIN1['STFT_bc']
dataInfo = sio.loadmat('data/f001_datainfo.mat')
BC_mean,BC_std = dataInfo['log_STFT_bc_mean'],dataInfo['log_STFT_bc_var']
AC_mean,AC_std = dataInfo['log_STFT_ac_mean'],dataInfo['log_STFT_ac_var']
# normalize data
BC = log_and_normalize(BC,BC_mean,BC_std)


# load model
LSTMModel.load_state_dict(torch.load('params.pkl'))

# start to evaluate
testnum = BC.shape[0]
result = []
for i in range(testnum):
    LSTMModel.batch_size = 1  # this should be write before hidden_init
    LSTMModel.hidden = LSTMModel.init_hidden() #
    DATA = BC[i]
    sequence = DATA.shape[0]
    DATA = DATA[np.newaxis,:,:]
    DATA = torch.from_numpy(DATA).float()
    DATA = Variable(DATA.cuda())
    predict = LSTMModel(DATA,[sequence])
    # tensor.cuda change back to numpy
    predict = predict.data.cpu().numpy()
    predict = predict.squeeze()
    # denormalize
    predict = de_log_and_normalize(predict,AC_mean,AC_std)
    result.append(predict)

sio.savemat('data/pytorch_lstm_one_to_one.mat',{'result':result})









