from __future__ import absolute_import

import os
import pickle

import numpy as np
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
# from skimage.transform import resize

import warnings

from keras import backend as K
from keras import activations, constraints, initializers, metrics, regularizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.engine import InputSpec, Layer
from keras.layers import Activation, BatchNormalization, Conv1D, concatenate, Dense, Input, LSTM, RNN, CuDNNLSTM, GRU, SimpleRNN, Masking, Reshape, multiply, GlobalAveragePooling1D, Permute, Dropout, Flatten
from keras.legacy import interfaces
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping


from IPython.display import SVG #jupyter notebook에서 보려고 
from keras.utils.vis_utils import model_to_dot # keras model을 dot language로 변환
from keras.utils import plot_model


import getpass
username = getpass.getuser()

# import tensorflow as tf
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

MAX_SEQUENCE_LENGTH = 31   # 획득된 데이터 길이; 우리조는 150으로 했음.
NB_CLASS = 7                # 지도학습에 이용될 데이터 클래스의 갯수; N,F,B,L,R,LW,RW 7개임.
CHANNEL = 5                 # 학습에 이용될 센서의 차원수; gx, gy, gz, ir1, ir2 5차원
datacnt = 100               # 사람&클래스 당 획득한 데이터 갯수; 우리조는 명당 200개*7클래스로 획득; 

test_size=0.1               # 학습을 위한 테스트셋 비율
epochs = 300                 # 딥러닝 학습 세대 수
batch_size = 32             # 배치 사이즈

who = 'bjksz'                # 모델별 이름을 다르게 하기 위한 변수; line (149), (146)을 보면 모델이름이 who에 의해 달라지는 것을 볼 수 있음.
# who = 'sz'                # 모델별 이름을 다르게 하기 위한 변수; line (149), (146)을 보면 모델이름이 who에 의해 달라지는 것을 볼 수 있음.

people = [x for x in who]  # 획득된 데이터 중, 여러 조합의 학습될 데이터를 고를 수 있는 옵션. 바로 위의 who가 들어가는 이유이기도 함.
data2jo = ["normal","forth","back","left","right","left wink","right wink"] # 7개 클래스
 
path = "C:/Users/"+username+"/desktop/amor/data"
folder_len_dict = {}
for folder_u in os.listdir(path):
    folder_len_dict[folder_u] = len(os.listdir(os.path.join(path, folder_u)))

def dataread(
    # datacnt=datacnt, 
    test_size=test_size, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, CHANNEL=CHANNEL):

    dataset = np.array([], dtype=np.int64).reshape(0,MAX_SEQUENCE_LENGTH,CHANNEL)
    yset = np.array([], dtype=np.int)

    for k in people:
        p=0
        for j in data2jo:
            tmp_datacnt=folder_len_dict.get(k+"_"+j)
            print(k,j,tmp_datacnt)
            for i in range(tmp_datacnt):
                if i==0:
                    os.chdir("C:/Users/"+username+"/desktop/amor/data/"+k+"_"+j)
                gxgygz1 = pandas.read_csv(j+' '+str(i+1)+".csv", header=None)
                gxgygz1 = gxgygz1.iloc[1:]
                # gxgygz1.loc[:3,:] = np.divide(gxgygz1.loc[:3,:],15)

                # if k=="h" and j!="sedentary at rest":      # 현성이 데이터 곱하기2.5
                #     gxgygz1=gxgygz1*2.5
                # if k=="h" and j=="sedentary at rest":      # but rest 곱하기 0.5
                #     gxgygz1=gxgygz1*0.2
                # gxgygz1.insert(gxgygz1.shape[1], "pattern", NB_CLASS-1) # pattern NB_CLASS-1
                if gxgygz1.shape[1] == (MAX_SEQUENCE_LENGTH+1):
                    gxgygz1 = gxgygz1.drop(gxgygz1.columns[0],axis=1)
                tg = np.transpose(gxgygz1.values).reshape(1,MAX_SEQUENCE_LENGTH,CHANNEL)
                dataset = np.vstack((dataset,tg))
                yset = np.append(yset,p) # pattern p
                # if k=='n' and j=='sedentary at rest':
                #     print(i)
            
            p = p+1

    x_train, x_test, y_train, y_test = train_test_split(dataset, yset, test_size=test_size,
        random_state=42, stratify=np.array(yset))

    return(x_train, y_train, x_test, y_test)

x_train, y_train, x_test, y_test = dataread()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# one-hot encoding
Y_train = to_categorical(y_train, num_classes=NB_CLASS)
Y_test = to_categorical(y_test, num_classes=NB_CLASS)

# 켈리브레이션
# X_train /= 255
# X_test /= 255
print(x_train[0])

X_train = x_train
X_test = x_test

# 객체 저장
os.chdir("C:/Users/"+username+"/desktop/amor/data_total_and_model")
data_total = [X_train, Y_train, X_test, Y_test]
with open('data_total_'+who+'_'+str(datacnt)+'.pkl', 'wb') as f:
    pickle.dump(data_total, f)

# # 객체 로드 
os.chdir("C:/Users/"+username+"/desktop/amor/data_total_and_model")
with open('./data_total_'+who+'_'+str(datacnt)+'.pkl', 'rb') as f:
    X_train, Y_train, X_test, Y_test = pickle.load(f)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

def squeeze_excite_block(input):
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se

def generate_lstmfcn(MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, NB_CLASS=NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(MAX_SEQUENCE_LENGTH, CHANNEL))
    # ip = Input(shape=(MAX_SEQUENCE_LENGTH, 1, CHANNEL))
    # x = Masking()(ip)
    # x = LSTM(NUM_CELLS)(x)
    x = LSTM(NUM_CELLS)(ip)
    x = Dropout(0.2)(x)

    # y = Permute((2, 1))(ip)
    # y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = Conv1D(32, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(64, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(32, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    # add load model code here to fine-tune

    return model

def get_callback(model_name, patient=20):
    ES = EarlyStopping(
        monitor='val_categorical_accuracy',  # Early Stopping을 어떤 수치를 보고 정할것인가
        patience=patient,  # 몇번 연속으로 val_loss가 개선되지 않았을때 Early Stopping 할 것인가
        mode='max',  # monitor하는 수치가 최소화되게? minimize objective function
        verbose=1  # 진행사항을 화면에 띄움
    )

    RR = ReduceLROnPlateau(
        monitor='val_categorical_accuracy',
        factor=0.5,  # learning rate를 줄이는 비율 / new_lr = lr * factor
        patience=5,
        min_lr=0.000001,  # learning_rate가 0.000001보다 작아지면 멈춤
        verbose=1,
        mode='max')

    MC = ModelCheckpoint(
        filepath=model_name,  # model file을 저장할 path
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True,  # 해당 epoch에서 가장 성능이 좋은 모델만 저장. 덮어쓰는 형식
        mode='max')

    return [ES, RR, MC]

model = generate_lstmfcn(MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,NB_CLASS=NB_CLASS,NUM_CELLS=8)
# print(model.summary())
# plot_model(model, to_file='C:/Users/"+username+"/desktop/amor/figure/keras_model.svg')
# SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# 모델 로드 # 계속 fitting
os.chdir("C:/Users/"+username+"/desktop/amor/data_total_and_model")
model = load_model('model_'+who+'_'+str(datacnt)+'.h5')


model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
              metrics=[metrics.categorical_accuracy])

# 모델 fitting
train_history = model.fit(X_train, Y_train, epochs=epochs, 
                          batch_size=batch_size, verbose=2, 
                          # validation_data=(X_test, Y_test),
                          validation_split=2/9,
                          callbacks=get_callback('model_'+who+'_'+str(datacnt)+'.h5')
                          )

# 모델 저장
os.chdir("C:/Users/"+username+"/desktop/amor/data_total_and_model")
model.save('model_'+who+'_'+str(datacnt)+'.h5')

# ######
# print(train_history.history.keys())  # ['acc', 'loss', 'val_acc', 'val_loss']  

# plt.figure(1)
# # summarize history for accuracy     
# plt.subplot(211)  
# plt.plot(train_history.history['categorical_accuracy'])  
# plt.plot(train_history.history['val_categorical_accuracy'])  
# plt.title('model accuracy')  
# plt.ylabel('accuracy')  
# plt.xlabel('epoch')  
# plt.legend(['train', 'test'], loc='upper left') 

# # summarize history for loss  
# plt.subplot(212)  
# plt.plot(train_history.history['loss'])  
# plt.plot(train_history.history['val_loss'])  
# plt.title('model loss')  
# plt.ylabel('loss')  
# plt.xlabel('epoch')  
# plt.legend(['train', 'test'], loc='upper left')  
# plt.show()  
# #######

# predict & evaluate
# Y_train_pred = (model.predict(X_train) > 0.5).astype('int64')
# Y_test_pred = (model.predict(X_test) > 0.5).astype('int64')
Y_train_pred = (to_categorical(np.argmax(model.predict(X_train),axis=1)))
Y_test_pred = (to_categorical(np.argmax(model.predict(X_test),axis=1)))
# score = model.evaluate(X_test, Y_test, batch_size=batch_size)
# print(score)
print("train: {}, test: {}".format(
    accuracy_score( Y_train, Y_train_pred.astype("int64")), 
    accuracy_score( Y_test, Y_test_pred.astype("int64"))
))
from sklearn.metrics import confusion_matrix
np.where(Y_test)
print(confusion_matrix(np.where(Y_test)[1], np.where(Y_test_pred)[1]))

# 결과 테이블
# submit_df = pandas.DataFrame({'Id':range(1,1+len(X_test)), 
#               'Solution':(model.predict(X_test) > 0.5).astype('int64').ravel()
#              }) # astype().ravel() 요거ㅏ하면 8(or 7) * 6 이되버림.
# submit_df.to_csv('submit_df', index=False)    
# print("complete")

# save SVG
# os.chdir("C:/Users/"+username+"/desktop/amor/figure")
# print(train_history.history.keys())
# x = range(0, len(train_history.history['val_loss']))
# train_categorical_accuracy_lst = train_history.history['categorical_accuracy']
# val_categorical_accuracy_lst = train_history.history['val_categorical_accuracy']
# plt.figure(figsize=(12, 4))
# plt.plot(x, train_categorical_accuracy_lst, label='train_score')
# plt.plot(val_categorical_accuracy_lst, label='val_score')
# plt.legend()
# plt.savefig("submit_plot3.svg")
# plt.show()

# save JPG
# def plot_history(history, title):
#     # list all data in history
#     print(history.history.keys())
#     # summarize history for accuracy
#     plt.plot(history.history['categorical_accuracy'])
#     plt.plot(history.history['val_categorical_accuracy'])
#     plt.title('Accuracy: '+title)
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
#     # summarize history for loss
#     plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
#     plt.title('Loss: '+title)
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.savefig("title"+".jpg") 

# plot_history(train_history,"C:/Users/"+username+"/desktop/amor/figure/test,jpg")
