import extract
import numpy as np
import serial
import threading

import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from drawnow import *
import timeit
import time

color_map = list({name for name in mcd.CSS4_COLORS if "xkcd:" + name in mcd.XKCD_COLORS})
exitThread = False

low = -10000
high = 10000


class Receiver:
    def __init__(self, *args, wide=31, show=15):
        self.port_num = ""
        self.baud_rate = 9600
        self.line = []
        self.cnt = 0
        self.pre = 0
        self.now = 0
        self.elapsed_time = []
        self.signal = None
        self.num_receive = 0
        self.num_sensors = 0
        self.dataArr = []
        self.Arr = None
        self.sensor_dict = dict(zip(args, np.zeros(len(args))))
        self.sensor_data = []
        self.wide = wide
        self.show = show
        self.low = -10000
        self.high = 10000
        self.start_time = timeit.default_timer()

    def set_port_num(self, port_num):
        self.port_num = port_num

    def set_baud_rate(self, baud_rate):
        self.baud_rate = baud_rate

    def set_using_sensor(self, *args):
        for sensor in [arg for arg in args if arg in self.sensor_dict.keys()]:
            self.sensor_dict[sensor] = 1

    def get_used_sensor(self):
        return [key for (key, val) in self.sensor_dict.items() if val == 1]
    
    def receive(self, ser, ignore_line=2):
        global exitThread

        self.start_time = timeit.default_timer()
        while not exitThread:
            for val in ser.read():
                # print(self.line)
                self.line.append(chr(val))
                if val == 10:
                    self.Arr = self.parsing_data(self.line)
                    print(self.Arr)
                    self.sensor_data.append(self.Arr)
                    if len(self.sensor_data) > self.wide:
                        self.sensor_data.pop()
                        self.elapsed_time.pop()
                    self.receive_time = timeit.default_timer()
                    self.elapsed_time.append(self.receive_time - self.start_time)
                    self.line = []

                    self.num_receive+=1

                if self.num_receive > self.wide:
                    if self.num_receive%self.show == 0:
                        drawnow(self.make_fig)
                        plt.pause(.000001)

    def make_fig(self):
        global color_map
        plt.title("title")
        cmap_list = color_map[:len(self.sensor_data)]
        plt.plot(self.sensor_data, color=cmap_list)
        plt.ylim(self.low, self.high)  # Set y min and max values
        plt.grid(True)
        # graph_legend = 
        legend = self.get_used_sensor()
        plt.legend(legend)

    # fig, ax = plt.subplots(figsize=(20,12))
    # fig = plt.plot(self.Arr, color=cmap_list)
    # ax.set_xticklabels(labels = elapsed_time)
    # ax.legend(graph_legend)
                        
    # def receive(self, ser, ignore_line=1):
    #     while TRUE:
    #         if self.num_receive <= ignore_line:
    #             self.signal = ser.readlines()
    #             self.num_receive += 1
    #             time.sleep(0.1)
    #             # del self.signal
    #         else:
    #             # self.signal = ser.readline()
    #             # print(self.signal)
    #             # time.sleep(0.001)
    #             receive_time = timeit.default_timer()
    #             self.elapsed_time.append(self.start_time - receive_time)
    #             # print(ser.readline())
    #             for val in ser.read():
    #                 self.line.append(chr(val))
    #                 if val == 10:
    #                     self.Arr = self.parsing_data(self.line)
    #                     print(self.Arr)
    #                     self.sensor_data.append(self.Arr)

    #                     if len(self.sensor_data) > self.wide:
    #                         self.sensor_data.pop()
    #                         self.elapsed_time.pop()
    #                     self.num_receive += 1
    #                     if self.num_receive % 100 == 0:
    #                         print(self.num_receive)

    #                     if receiver.num_receive > 200:
    #                         drawnow(make_fig(array=receiver.sensor_data, legend=receiver.get_used_sensor()))
    #                         stop_receive = get_stop_situation()

    def parsing_data(self, data, strip='[]', split=','):
        self.line = "".join(data)
        self.dataArr = self.line.strip().strip(strip).split(split)
        self.dataArr = np.array([int(val) for val in self.dataArr])
        return self.dataArr[[idx for idx, (k, v) in enumerate(self.sensor_dict.items()) if v == 1]]


def set_graph_options(low_, high_):
    global low
    global high

    low = low_
    high = high_






def get_stop_situation():
    return extract.stop_listener


def get_data_folder():
    return extract.data_folder_path


def get_pattern():
    return extract.pattern


if __name__ == "__main__":
    print("Start")
    print(get_data_folder())
    #Receiver = Receiver("a", "b", "c", "d")
    #Receiver.set_using_sensor("a", "c")
    # extract.set_data([1, 2, 3])
    # extract.set_extract_options(pattern_name="foward")
    # print(path, pattern, beep1, beep2, elapsed, start_index)

"""
    stop_receive = get_stop_situation()
    while not stop_receive:
        Receiver.receive()
        with extract.Listener(on_press=extract.on_press) as listener:
            while not extract.stop_listener:
                listener.join()
            stop_receive = get_stop_situation()

    print("Program Done...")
"""


