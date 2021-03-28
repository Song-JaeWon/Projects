import os
os.chdir("c:/users/qldh1/desktop/amor")

from receive import *
import extract
import training
import serial
import signal

from pynput.keyboard import Listener

exitThread = False

def run():
    print("Select Process:")
    print("1. Data Receive \n2. Read Data and Compile Model\n3. Test Score\n4. Predict")
    num = int(input("Select Num : "))
    if num == 1:
        signal.signal(signal.SIGINT, handler)

        receiver = Receiver("1","2","3","4","5","6","7","8")
        receiver.set_using_sensor("4","5","6","7","8")

        # receiver.set_port_num("COM7")
        # receiver.set_baud_rate(9600)
        """
        wide = input("wide[Default=100] : ")
        show = input("Frequency[Default=10] : ")
        low = input("Y Limitation - below [Default=-10000]: ")
        high = input("Y Limitation - upper [Default=10000]: ")
        graph_options = [wide, show, low, high]

        receiver.set_graph_option()
        """
        print(receiver.get_used_sensor())
        print("Receiving Data...")
        stop_receive = get_stop_situation()
        ser = serial.Serial("COM11", 9600, timeout=0)
        time.sleep(0.1)
        ser.readline()
        time.sleep(0.1)
        ser.readline()

        thread_serial = threading.Thread(target=receiver.receive, args=(ser,))
        thread_serial.start()

        # stop_receive = get_stop_situation()

        print("Successfully Received!")

    elif num == 2:
        part_2 = training.Training()
        train_X, test_X, train_y, test_y = part_2.split()
        model = part_2.get_model()
        history = training.compile_model(model)
        history = training.training_model(history, train_X, train_y, validation_split=2/9)
        history.summary()

        training.get_learning_curve(history)

    elif num == 3:
        print("Predict")


def handler(signum, frame):
    global exitThread
    exitThread = True

if __name__ == "__main__":
    run()


# with pynput.keyboard.Listener(on_press=extract.on_press) as listener:
#     while not extract.stop_listener:
#         listener.join()