import sys
import cv2
import time


class Camera:
    # constructor
    x_coord = 0
    y_coord = 0
    fps = 0

    human_head_mm = 560

    def __init__(self, object_cascade, object2_cascade, camera_port):
        """
        vision class init constructor.

        constructs the vision class and allows the functions to be called. feed is meant to represent the camera being used.
        object_cascade is the string representation of the haar cascade being used by the CascadeClassifier. Upon execution
        the class returns a x,y coordinate representation of the box that was drawn around the object.

        :param self:
        :param object_cascade: String, path to xml file of haarcascade.
        :param camera_port: camera object
        :return:
        """
        self.camera_port = camera_port
        self.feed = cv2.VideoCapture(self.camera_port)
        self.object_cascade = object_cascade
        self.object2_cascade = object2_cascade

    def camera_ready(self):
        """
        camera_ready "turns on the camera, returns a true or false value for the camera open function, 0 is default for a
        built-in laptop webcam, will change if different for rbPI"
        :param self:
        :param feed: camera object
        :return: boolean representation of camera on.
        """
        return self.feed.isOpened()

    def kill_feed(self):
        # When everything's done, release the video capture object
        self.feed.release()

        # Closes all the frames
        cv2.destroyAllWindows()

    def get_frames_per_sec(self):
        # Number of frames to capture
        num_frames = 120

        print("Capturing {0} frames".format(num_frames))

        # Start time
        start = time.time()

        # Grab a few frames

        # End time
        end = time.time()

        # Time elapsed
        seconds = end - start
        print("Time taken : {0} seconds".format(seconds))

        # Calculate frames per second
        fps = num_frames / seconds
        print("Estimated frames per second : {0}".format(fps))

    def detect_object(self):
        """
        detect object - method detects an object given the specified object perameters given by the haar cascade and sends
        x and y coordinates to the class attributes. method is complete when feed is killed.
        :param self:
        :return:
        """
        front_face_cascade = cv2.CascadeClassifier(self.object_cascade)
        profile_face_cascade = cv2.CascadeClassifier(self.object2_cascade)

        while self.feed:
            ret, feed_by_frame = self.feed.read()
            gray_feed = cv2.cvtColor(feed_by_frame, cv2.COLOR_BGR2GRAY)

            objects = front_face_cascade.detectMultiScale(gray_feed, 1.2, 5)
            for (x, y, w, h) in objects:
                cv2.rectangle(feed_by_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.x_coord = (x + (x + w)) / 2
                self.y_coord = (y + (y + h)) / 2
                # print(x_coord, y_coord)

            objects_2 = profile_face_cascade.detectMultiScale(gray_feed, 1.2, 5)
            for (x, y, w, h) in objects_2:
                cv2.rectangle(feed_by_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                self.x_coord = (x + (x + w)) / 2
                self.y_coord = (y + (y + h)) / 2
                # print(x_coord, y_coord)

            # for debug, remove later
            cv2.imshow('Frame', feed_by_frame)
            cv2.imshow('Frame2', gray_feed)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.kill_feed()


front_face_objects = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
profile_face_objects = cv2.data.haarcascades + 'haarcascade_profileface.xml'

camera = Camera(front_face_objects, profile_face_objects, 0)
camera_ready = camera.camera_ready()
camera.detect_object()

