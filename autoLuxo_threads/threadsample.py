# thread sample
from Computer_Vision.vision import vision
from multiprocessing import Process
import os
import time


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def f(name):
    info('function f')
    print('hello', name)


def run_camera(camera_port):
    camera = vision.Camera(camera_port)
    camera.run_camera()


def collect_user_data(name):
    for x in range(10):
        info('function collect_user_data')
        time.sleep(1)
        print("Hello", name)


if __name__ == '__main__':

    info("Console")
    p = Process(target=run_camera, args=(0,))
    q = Process(target=collect_user_data, args=('ian',))
    q.start()
    p.start()
    p.join()
    q.join()






