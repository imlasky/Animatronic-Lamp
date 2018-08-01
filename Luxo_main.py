import warnings
warnings.filterwarnings("ignore")
from flask import Flask
from flask_ask import Ask, statement, convert_errors
from threading import Thread

import logging
import cv2
import vision
import gestures
import servos
import os

app = Flask(__name__)
ask = Ask(app, '/')

logging.getLogger("flask_ask").setLevel(logging.DEBUG)
threads = []
feed = cv2.VideoCapture(0)
servo0 = servos.Servo(250, 600, 0)
servo1 = servos.Servo(527, 225, 1)
servo2 = servos.Servo(154, 382, 2)
servo3 = servos.Servo(105, 620, 3)
light_flag = False 

gestures.home([servo0, servo1, servo2, servo3])

gestures.wake_up([servo1, servo2, servo3])

@ask.intent('LightOnOff', mapping={'OnOff': 'state'})
def turn_on_off(state):
    global light_flag
    if state == 'on': 
        if light_flag:
            return statement('Are you blind?')
        else:
            light_flag = not light_flag 
            return statement('Turning light {}'.format(state))
    elif state == 'off':
        if light_flag:
            light_flag = not light_flag 
            return statement('Turning light {}'.format(state))
        else:
            return statement('The light is already off.')

@ask.intent('DontLookAtObject', mapping={'subject': 'subject'})
def dont_look_at_object(subject):
    if len(threads) > 0:
        threads[0].join()
        del threads[0]
    return statement('Okay')

@ask.intent('LookAtObject', mapping={'subject': 'subject'})
def look_at_object(subject):
    if len(threads) > 0:
        threads[0].join()
        del threads[0]
    if subject == 'me':
        t = vision.Camera(feed, 0)
        threads.append(t)
        t.start()
        return statement('Looking at you')
    if subject == 'this':
        t = vision.Camera(feed, 2)
        threads.append(t)
        t.start()
        return statement('Looking for it')
    if subject == 'my desk':
        t = vision.Camera(feed, 2)
        threads.append(t)
        t.start()
        return statement('Looking at your desk')

if __name__ == "__main__":
    light_flag = 0
    app.run()

