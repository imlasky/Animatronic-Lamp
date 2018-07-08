# thread sample
import threading
import time
from autoLuxo_threads import sampleClass
from Computer_Vision.vision import vision

exitFlag = 0


# class ThreadSample(threading.Thread):
#
#     def __init__(self, threadID, name, counter):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.name = name
#         self.counter = counter
#
#     def run(self):
#         print("Starting" + self.name)
#         self.print_time(self.name, 5, self.counter)
#         print("Exiting " + self.name)
#
#     def print_time(self, name, counter, delay):
#         while counter:
#             if exitFlag:
#                 name.exit()
#             time.sleep(delay)
#             print("%s: %s" % (name, time.ctime(time.time())))
#             counter -= 1

def print_object_coord():
    print(cam.get_object_coord())


if __name__ == '__main__':

    cam = vision.Camera(0)
    cam.start()

    print("exit")
