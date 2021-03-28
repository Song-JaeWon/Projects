# 분류하고자 하는 패턴 수
# 사용될 센서의 수
# 데이터 저장되어 있는 폴더경로

# 데이터 읽어올 함수
# 모델 generate 함수
# 모델 저장함수
# 모델 load 함수

# 학습곡선 graph function



import pickle
import receive
import extract
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Tensorflow 2.0 version 사용 시 
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Conv1D, Dense, Input, LSTM, concatenate, \
    GlobalAveragePooling1D, Permute, Dropout, Masking, Reshape, multiply, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Tensorflow 1.x version 사용 시
"""
from keras import backend as K
from keras import metrics
from keras.models import Sequential, Model, load_model
from keras.layers import Activation, Conv1D, Dense, Input, LSTM, concatenate, \
    GlobalAveragePooling1D, Permute, Dropout, Masking, Reshape, multiply, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
"""

from sklearn.model_selection import train_test_split

version = 1
epochs = 50
batch_size = 32

class Training:
    def __init__(self):
        self.num_patterns = 7
        self.MAX_SEQ = 31   # number of timesteps
        self.num_sensors = 6

        # file_list = os.listdir(path)
        # pattern_list = set([re.findall('\\\\([a-zA-Z]*)_', file)[-1] for file in file_list])

    def read_data(self):
        path = get_data_folder()
        pattern_names = os.listdir(path)
        if not pattern_names:
            return print("Empty!")
        data_set = np.array([], dtype=np.int64).reshape(0, self.MAX_SEQ, self.num_sensors)
        y_set = np.array([], dtype=np.int)
        for pattern in pattern_names:
            if not os.path.exists(os.path.join(path, pattern)):
                print("{} folder doesn't exist. Check Again!".format(pattern.upper()))

        try:
            for i, pattern in enumerate(pattern_names):
                print("Reading Data in {} folder...".format(pattern))
                PATH = os.path.join(path, pattern)
                data_cnt = len(os.listdir(PATH))
                data_list = os.listdir(PATH)
                os.chdir(PATH)

                for j in range(data_cnt):
                    tmp_data = pd.read_csv(data_list[j], header=None)
                    tmp_data = tmp_data.values.reshape(1, self.MAX_SEQ, self.num_sensors)
                    data_set = np.vstack((data_set, tmp_data))
                    y_set = np.append(y_set, i)
                print("Reading Data Completed!")
        except FileNotFoundError:
            print("File doesn't exist!")
        return data_set, y_set

    def split(self, test_size=0.1):
        data, label = self.read_data()
        print("Splitting Data...")
        train_X, test_X, train_y, test_y = train_test_split(data, label, test_size=test_size,
                                                            random_state=42, stratify=np.array(label))
        train_y = to_categorical(train_y, num_classes=self.num_patterns)
        test_y = to_categorical(test_y, num_classes=self.num_patterns)

        print("Splitting Data Done!")
        return train_X, test_X, train_y, test_y

    def squeeze_excite_block(self, layers, num_filter):
        se = GlobalAveragePooling1D()(layers)
        se = Reshape((1, num_filter))(se)
        se = Dense(num_filter // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(num_filter, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        se = multiply([layers, se])
        return se

    def get_model(self, MAX_SEQ=31, num_sensors=6, num_patterns=1, num_cells=8):
        print("Getting Model")
        ip = Input(shape=(MAX_SEQ, num_sensors))
        # x = Masking()(ip)
        x = LSTM(num_cells)(ip)
        x = Dropout(0.2)(x)

        # y = Permute((2, 1))(ip)
        # y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        filter_1 = 64
        filter_2 = 128
        y = Conv1D(filter_1, 8, padding='same', kernel_initializer='he_uniform')(ip)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y, num_filter=filter_1)

        y = Conv1D(filter_2, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y, num_filter=filter_2)

        y = Conv1D(filter_1, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        out = Dense(num_patterns, activation='softmax')(x)

        model = Model(ip, out)
        print("Getting Model Done")
        return model


def compile_model(model, learning_rate=0.0001):
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(lr=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-8),
                  metrics=[metrics.categorical_accuracy])
    return model


def save_model(model):
    global version

    PATH = os.path.join(extract.path, 'models')
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    model_name = os.path.join(PATH, "model_v{}.h5".format(version))

    if not os.path.exists(model_name):
        model.save(model_name)
        version += 1
    else:
        print("Same filename already exists. Overwrite it?")
        overwrite = input("Y/N")
        if overwrite in ["Y", "y"]:
            model.save(model_name)
            version += 1
        elif overwrite in ["N", "n"]:
            version += 1
        else:
            print("Wrong Keyword")
            pass


def get_data_folder():
    return extract.data_folder_path


def training_model(history, X, y, epochs=epochs, batch_size=batch_size, validation_split=2/9, verbose=1):
    global version

    PATH = os.path.join(extract.path, 'models')
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    model_name = os.path.join(PATH, "model_v{}.h5".format(version))
    if not os.path.exists(model_name):
        history.save(model_name)
        version += 1

    history.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose,
              callbacks=get_callback(model_name))
    return history


def get_callback(model_name, patient=10):
    ES = EarlyStopping(
        monitor='val_categorical_accuracy',  # Early Stopping을 어떤 수치를 보고 정할것인가
        patience=patient,  # 몇번 연속으로 val_loss가 개선되지 않았을때 Early Stopping 할 것인가
        mode='min',  # monitor하는 수치가 최소화되게? minimize objective function
        verbose=1  # 진행사항을 화면에 띄움
    )

    RR = ReduceLROnPlateau(
        monitor='val_categorical_accuracy',
        factor=0.5,  # learning rate를 줄이는 비율 / new_lr = lr * factor
        patience=3,
        min_lr=0.000001,  # learning_rate가 0.000001보다 작아지면 멈춤
        verbose=1,
        mode='min')

    MC = ModelCheckpoint(
        filepath=model_name,  # model file을 저장할 path
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True,  # 해당 epoch에서 가장 성능이 좋은 모델만 저장. 덮어쓰는 형식
        mode='min')

    return [ES, RR, MC]


def get_model(model_name):
    return load_model(model_name)


def get_learning_curve(history):
    plt.figure(1)
    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    # print(get_folder_path())
    A = Training()
    model = A.get_model()
    model.summary()