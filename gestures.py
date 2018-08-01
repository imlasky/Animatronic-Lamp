import library.servos as servos
import time

def shake_head(servo):

    servo.set_pos(-0.75)
    time.sleep(0.25)
    servo.set_pos(0.75)
    time.sleep(0.25)
    servo.set_pos(-0.25)
    time.sleep(0.25)
    servo.set_pos(0.25)
    time.sleep(0.25)
    servo.set_pos(0)

def wake_up(servo_list):

    # Actually servos 1, 2, and 3, but fuck ya numbering
    servo_list[0].set_pos(0.5)
    servo_list[1].set_pos(0.5)
    shake_head(servo_list[2])

def home(servo_list):

    servo_list[0].set_pos(0)
    servo_list[1].set_pos(0)
    servo_list[2].set_pos(0)
    servo_list[3].set_pos(0)


    
