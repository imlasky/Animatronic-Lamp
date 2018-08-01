import Adafruit_PCA9685
import numpy as np

pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)


class Servo():

    def __init__(self, lower_bound, upper_bound, pin):

        self.center = (upper_bound + lower_bound)/2

        self.offset = upper_bound - self.center

        self.pin = pin

    def set_pos(self, pos):

        pwm_set = int(self.center) + int(pos * float(self.offset)) 

        pwm.set_pwm(self.pin, 0, pwm_set)

        
    def set_servo_pulse(self, channel, pulse):
        pulse_length = 1000000 
        pulse_length //= 60   	
        print('{0}us per period'.format(pulse_length))
        pulse_length //= 4096
        print('{0}us per bit'.format(pulse_length))
        pulse *= 1000
        pulse //= pulse_length
        print(pulse)
