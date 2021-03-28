import os
import serial
import time
import timeit
import datetime
import winsound as ws
import csv

import signal
import threading
import matplotlib.pyplot as plt
from drawnow import *
from pynput.keyboard import Key, Listener

from multiprocessing import Process
import sys
import numpy as np

# 주의 키보드는 바로 눌렀다 땔 것

#####################################################################################
import getpass
username = getpass.getuser()
os.chdir("C:/Users/"+username+"/desktop/amor/data") # 파일이 저장될 위치

port = 'COM8'    # 시리얼 포트
baud = 9600    # 시리얼 보드레이트(통신속도)
low = -100         # y축 최솟값
high = 1100      # y축 최댓값
wide = 31       # 그래프 폭 결정
show = 4         # 그래프가 업데이트되는 단위; 작을 수록 그래프가 빠르게 재출력되지만 컴퓨터 선능이 떨어지면 딜레이가 발생하게 되니 적절할 값을 선택할 것; 출력하는 뽑는 변수(ax, mg, amp1 등)에 따라서도 달라짐.
beep1 = 1000     # beep음정 높을수록 고음
beep2 = 300      # 소리 길이 단위(ms); 이 소리가 나올때 csv파일로 저장됨.
title = "normal"  # 파일명, 그래프 타이틀
index = 557          # 데이터 넘버링(index-1)

flag_ax = False
flag_ay = False
flag_az = False
flag_gx = True
flag_gy = True
flag_gz = True
flag_ir1 = True
flag_ir2 = True

graph_legend=['gx','gy','gz', 'ir1', 'ir2']

### + plot 도 주석 바꿔주기

#####################################################################################

line = [] #라인 단위로 데이터 가져올 리스트 변수
cnt=0
count=0
stop=0.0
pre_time=0
now_time=0

now=''
timex = []

ax = []
ay = []
az = []
gx = []
gy = []
gz = []
ir1 = []
ir2 = []

exitThread = False   # 쓰레드 종료용 변수

def on_press(key):
    global now_time
    global pre_time

    now_time=time.time()

    if (now_time-pre_time)>1:
        savecsv()
        pre_time=time.time()
        print('{0} pressed. {1}.csv에 저장되었습니다. '.format(key, now))

def savecsv():
    global now
    global index

    now = str(datetime.datetime.now())[11:19]
    now=now.replace(":","-")
    now=now.replace(" ","_")
    index = index + 1
    
    # 이걸 바꾸면 파일명을 바꿔줌
    now = title + ' ' + str(index-1)

    f = open(now+'.csv', 'w', encoding='utf-8', newline='')
    ws.Beep(beep1,beep2) #  음정, 시간(ms)
    wr = csv.writer(f)
    wr.writerow(timex)
    if flag_ax:
        wr.writerow(ax)
    if flag_ay:
        wr.writerow(ay)
    if flag_az:
        wr.writerow(az)
    if flag_gx:
        wr.writerow(gx)
    if flag_gy:
        wr.writerow(gy)
    if flag_gz:
        wr.writerow(gz)
    if flag_ir1:
        wr.writerow(ir1)
    if flag_ir2:
        wr.writerow(ir2)
    f.close()  

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

#쓰레드 종료용 시그널 함수
def handler(signum, frame):
    exitThread = True

#데이터 처리할 함수
def parsing_data(data):
    tmp = ''.join(data)
    # print(tmp)                       # 이거 주석 풀면 데이터 전부 출력 됨.
    dataArray = tmp.strip().strip('[]').split(',')
    # print(dataArray)    
    return(dataArray)
    
#본 쓰레드
def readThread(ser):
    global line
    global exitThread
    global timex
    global stop
    global cnt

    global ax
    global ay
    global az
    global gx
    global gy
    global gz
    global ir1
    global ir2

    start = timeit.default_timer()

    # 쓰레드 종료될때까지 계속 돌림
    while not exitThread:
        #데이터가 있있다면
        for c in ser.read():
            #line 변수에 차곡차곡 추가하여 넣는다.
            line.append(chr(c))
            # print(cnt)

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

                #line 변수 초기화
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

    if cnt % 5000:
        gc.collect()

if __name__ == "__main__":
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

    #시리얼 읽을 쓰레드 생성
    threadserial = threading.Thread(target=readThread, args=(ser,))
    threadserial.start()


# 아무 키나 누르면 'on_press()'' 함수가 실행됨
with Listener(
        on_press=on_press) as listener:
    listener.join()