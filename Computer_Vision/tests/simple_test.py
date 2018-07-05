import Computer_Vision.vision.vision as vision

"""
    vision.Camera instance args
"""

camera = vision.Camera(0)                       # load camera instance
vision.camera_ready = camera.camera_ready()     # ready camera
camera.detect_object('front_face')              # choose thing to look at
