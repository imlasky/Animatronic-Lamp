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
    camera = vision.Camera(0)
    num = Value('d', 0.0)
    arr = Array('i', range(2))

    p = Process(target=camera.detect_object('front_face'))
    p.start()
    print(camera.get_object_coord())    
    p.join()







