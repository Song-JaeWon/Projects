import numpy as np
import time
import csv
import winsound as ws
import os
import receive

from pynput.keyboard import Key, Listener

start_index = 1
curr_time = 0
pre_time = 0
beep1 = 1000
beep2 = 300
path = os.getcwd()      # 상위폴더 / '데이터저장폴더', '모델저장폴더'를 하위 폴더로 가짐
data_folder_path = os.path.join(path, "patterns")   # 데이터저장폴더 / 각 패턴별 폴더를 하위 폴더로 가짐
pattern = "pattern"
sensor_data = np.array([])
stop_listener = False
elapsed = 1


def save_csv(folder_path=data_folder_path, title=pattern, data=sensor_data):
    global start_index

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print("Folder Created name as 'patters'")   # 'patters'폳더가 없으면 생성

    PATH = os.path.join(folder_path, title)

    if not os.path.exists(PATH):
        os.mkdir(PATH)     # 각 pattern 폴더가 존재하지 않으면 생성
        print("Folder Created name as {}".format(title))

    file_name = os.path.join(PATH, title+"_"+str(start_index)+'.csv')
    f = open(file_name, 'w', encoding='utf-8', newline='')
    ws.Beep(beep1, beep2)
    wr = csv.writer(f)
    wr.writerow(data)
    f.close()
    start_index += 1
    print("File saved as {}".format(file_name))
    return file_name


def on_press(pressed_key):
    global curr_time
    global pre_time
    global stop_listener
    global elapsed
    global path
    global pattern
    global sensor_data

    curr_time = time.time()

    if (curr_time - pre_time) > elapsed:
        if pressed_key == Key.enter:
            file_name = save_csv(data_folder_path, pattern, sensor_data)
            pre_time = time.time()
            return print("{0} key pressed.".format(str(pressed_key).split(".")[1].upper()))
        elif pressed_key == Key.esc:
            stop_listener = True
            ws.Beep(beep1, beep2)
            print("{} key pressed.\nStop Listener".format(str(pressed_key).split(".")[1].upper()))
            return False
        else:
            pass


def set_data(data=sensor_data):
    global sensor_data

    sensor_data = data


def set_extract_options(folder_path=os.getcwd(), pattern_name="pattern", beep_sound=1000,
                        beep_length=300, index=1, elapsed_time=1):
    global path
    global pattern
    global start_index
    global beep1
    global beep2
    global elapsed

    path = folder_path
    pattern = pattern_name
    start_index = index
    beep1 = beep_sound
    beep2 = beep_length
    elapsed = elapsed_time
    return None


if __name__ == "__main__":
    print(data_folder_path)