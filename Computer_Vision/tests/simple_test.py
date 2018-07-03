import sys
import cv2
import Computer_Vision.vision.vision as vision

front_face_objects = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
profile_face_objects = cv2.data.haarcascades + 'haarcascade_profileface.xml'

camera = vision.Camera(0)
vision.camera_ready = camera.camera_ready()
camera.detect_object('profile_face')
