# thread sample
from Computer_Vision.vision import vision
from multiprocessing import Process, Value, Array
import os
import time


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def f(n, a):
    info('f')
    n.value = 3.1415927
    for i in range(len(a)):
        a[i] = -a[i]


if __name__ == '__main__':
    """Main method"""
    camera = vision.Camera(0)           # starts camera class
    num = Value('d', 0.0)               # this was something i was trying from the tutorial
    arr = Array('i', range(2))          # this too

    p = Process(target=camera.detect_object('front_face'))  # classify process as detect object
    p.start()                                               # start process
    print(camera.get_object_coord())                        # print coords from main process
    p.join()                                                # merge process







