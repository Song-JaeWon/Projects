from __future__ import absolute_import

import serial
import time
import timeit
import datetime
import csv
import signal
import threading
from pynput.keyboard import Key, Listener
from multiprocessing import Process
import sys

import os
import pickle
import numpy as np
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from drawnow import *

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import train_test_split
# from skimage.transform import resize

import warnings

from keras import backend as K
from keras import activations, constraints, initializers, metrics, regularizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.engine import InputSpec, Layer
from keras.layers import Activation, BatchNormalization, Conv1D, concatenate, Dense, Input, LSTM, RNN, CuDNNLSTM, GRU, SimpleRNN, Masking, Reshape, multiply, GlobalAveragePooling1D, Permute, Dropout, Flatten
from keras.legacy import interfaces
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

from IPython.display import SVG #jupyter notebook에서 보려고 
from keras.utils.vis_utils import model_to_dot # keras model을 dot language로 변환
from keras.utils import plot_model

import gc

from pygame import mixer  # 소리
import getpass
username = getpass.getuser()

MAX_SEQUENCE_LENGTH = 31
CHANNEL = 5

#####################################################################################
port = 'COM8'    # 시리얼 포트
baud = 9600    # 시리얼 보드레이트(통신속도)
wide = 31       # 그래프 폭ㄲ 결정
show = 10
low = -100
high = 1100

flag_ax = False
flag_ay = False
flag_az = False
flag_gx = True
flag_gy = True
flag_gz = True
flag_ir1 = True
flag_ir2 = True

graph_legend=['gx','gy','gz','ir1','ir2']
#####################################################################################

line = [] #라인 단위로 데이터 가져올 리스트 변수
cnt=0
pre_time=0
now_time=0
stop=0.0

timex = []

ax = []
ay = []
az = []
gx = []
gy = []
gz = []
ir1 = []
ir2 = []

list_pred = []
exitThread = False   # 쓰레드 종료용 변수

# 학습 모델 로드 
from keras.models import *
from keras.utils import *
# import tensorflow as tf
import os
os.chdir("C:/Users/"+username+"/desktop/amor/data_total_and_model")
who = "bjksz"
datacnt = 100
title = "Classifying"+'model'+'_'+who
model = load_model('model_'+who+'_'+str(datacnt)+'.h5')

# 모델을 로드할 때 서로 같은 그래프를 공유하려고 하여 발생한 이슈었습니다.
# Keras TF 백엔드에선 전역변수로 Tensorflow Session을 가지고 있고, 해당 세션에 바인드 되어있는 그래프를 사용하여 연산을 정의하는것으로 보입니다.
# Keras가 세션을 전역변수로 가지고 있기에 각 Thread에서 서로 다른 Session을 가지게 할 수 없고, 그에 따라 모두가 같은 그래프를 공유하게 됩니다.
# 서로 다른 그래프를 사용할 수 없다는 가정하에 tf.name_scope()를 이용해 매번 새로운 네임스코프를 만들어 모델을 로드하는 방법을 시도했을 때 위 에러가 발생하지 않고 정상적으로 로드하였습니다.
# 하지만 application thread에서 모델이 predict하는 동안 model update thread에서 로드가 일어날 경우 graph에 동시에 접근하는 일이 발생이 되는데, 이 때 TF의 graph가 thread safe하지 않아 새로운 에러가 발생하게 됩니다.
# 결국 Keras를 multi thread 환경에서 운용하는것은 힘들다고 판단하고 TF high-level API를 이용하기로 결정했습니다.

print(time.time())

def parsing_data(data):
    tmp = ''.join(data)
    # print(tmp)                       # 이거 주석 풀면 데이터 전부 출력 됨.
    dataArray = tmp.strip().strip('[]').split(',')
    dataArray = np.array([int(x) for x in dataArray])
    # dataArray[:3] = dataArray[:3]/15
    return(dataArray)

# 실시간 차트 함
def makeFig(): 

    plt.title(title)                     #Plot the title

    plt.plot(
        # timex, ax, 'k-',
        # timex, ay, 'r-',
        # timex, az, 'k-',
        gx, 'r-',
        gy, 'g-',
        gz, 'b-',
        ir1, 'k--',
        ir2, 'o--',
        )
    
    plt.ylim(low,high)                #Set y min and max values
    plt.grid(True)   
    plt.legend(graph_legend)

def pred(pattern):
    data2jo = ["normal","forth","back","left","right","left wink","right wink"] # 7개 클래스
    onehot = to_categorical(np.argmax(model.predict(pattern),axis=1))
    arg = np.argmax(onehot) 
    what = data2jo[arg]
    print(what)
    return(what)

def real_pred(what_tmp):
    global list_pred
    global now_time
    global pre_time

    if (what_tmp!='normal'):
        now_time=time.time()

        if (now_time-pre_time)>1:

            list_pred.append(what_tmp)

            if len(list_pred) >= 4 :
                if len(set(list_pred[-4:]))==1:
                    print(list_pred[-1])

                    mixer.init()
                    mixer.music.load("c:/users/"+username+"/desktop/amor/print_voice/"+list_pred[-1]+".mp3")
                    mixer.music.set_volume(1)
                    mixer.music.play()
                    pre_time=time.time()
                    list_pred = []


#쓰레드 종료용 시그널 함수
def handler(signum, frame):
    exitThread = True

start = timeit.default_timer()

#종료 시그널 등록
signal.signal(signal.SIGINT, handler)

#시리얼 열기
ser = serial.Serial(port, baud, timeout=0)

# 처음의 두 줄 데이터 버림; 첫줄 데이터에 불량이 많아 자주 발생되는 에러를 방지.
signal = ser.readline()
del signal
time.sleep(0.1)
signal = ser.readline()
del signal

# 쓰레드 종료될때까지 계속 돌림
while not exitThread:

    #데이터가 있있다면
    for c in ser.read():
        #line 변수에 차곡차곡 추가하여 넣는다.
        line.append(chr(c))

        if c == 10: #라인의 끝을 만나면..
            #데이터 처리 함수로 호출
            Arr = parsing_data(line)
            if flag_ax:
                temp_ax = Arr[0]
                ax.append(int(temp_ax))
            if flag_ay:
                temp_ay = Arr[1]
                ay.append(int(temp_ay))
            if flag_az:    
                temp_az = Arr[2]
                az.append(int(temp_az))
            if flag_gx:
                temp_gx = Arr[3]
                gx.append(int(temp_gx))                                          
            if flag_gy:
                temp_gy = Arr[4]
                gy.append(int(temp_gy))
            if flag_gz:
                temp_gz = Arr[5]
                gz.append(int(temp_gz))
            if flag_ir1:
                temp_ir1 = Arr[6]
                ir1.append(int(temp_ir1))
            if flag_ir2:
                temp_ir2 = Arr[7]
                ir2.append(int(temp_ir2))

            stop=timeit.default_timer()
            timex.append(stop-start)

            del line[:]

            # 실시간 그래프 출력
            if cnt%show==0:
                drawnow(makeFig)
                plt.pause(.000001)

            cnt = cnt+1
            if(cnt > wide):     #If you have 'wide' or more points, delete the first one from the array
                if flag_ax:
                    ax.pop(0)
                if flag_ay:
                    ay.pop(0)
                if flag_az:
                    az.pop(0)
                if flag_gx:
                    gx.pop(0)
                if flag_gy:
                    gy.pop(0)
                if flag_gz:
                    gz.pop(0)
                if flag_ir1:
                    ir1.pop(0)
                if flag_ir2:
                    ir2.pop(0)

                timex.pop(0)

            if ((cnt%3==0) and (cnt>MAX_SEQUENCE_LENGTH)):
                prepred = np.transpose(np.vstack((gx,gy,gz,ir1,ir2))).reshape(1,MAX_SEQUENCE_LENGTH,CHANNEL)
                what_tmp = pred(prepred)
                real_pred(what_tmp)

                if cnt % 5000:
                    
                    gc.collect()