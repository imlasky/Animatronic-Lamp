import cv2
import numpy as np
import sys
import time
import tensorflow as tf
import hardware as hw
import threading


class Camera(threading.Thread):
    """
        Camera Class - class that takes control of the camera and its functionality. The objective of this class is to
    run the camera, identify objects of interest and specify the location based on the camera's position. The class will
    also be able to identify the objects velocity and acceleration in order to help with the movement of the device.
        Arguments: Camera accepts 1 argument with three options. "front_face", "profile_face", and "paper" in reference
        to the object that its trying to find. For options front_face and profile face
    """

    # global vars.
    x_coord = 0
    y_coord = 0
    x_vel = 0
    y_vel = 0
    x_acc = 0
    y_acc = 0
    curr_time_x = 0
    curr_time_y = 0

    center_poly = 0

    def __init__(self, feed, obj):
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
        self.object_coord = []
        self.object_vel = []
        self.object_acc = []

        self.front_face_objects = './haarcascades/haarcascade_frontalface_default.xml'
        self.profile_face_objects = './haarcascades/haarcascade_profileface.xml'

        #self.camera_port = camera_port
        #self.feed = cv2.VideoCapture(self.camera_port)
        self.feed = feed
        self.obj = obj
        self.flag = 1

        self.positions = [0, 0, 0, 0]
        self._stopevent = threading.Event( )
        threading.Thread.__init__(self)



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

    def run(self):

        self.detect_object(self.obj, self.feed)

    def join(self, timeout=None):

        print('join statement')
        self.flag = 0
        self._stopevent.set( )
        threading.Thread.join(self, timeout)

            #cv2.imshow('frame', frame)
            #if cv2.waitKey(1) & 0xff == ord('q'):
            #    break

    def detect_object(self, cascade, feed):
        """
        detect object - method detects an object given the specified object perameters given by the haar cascade and sends
        x and y coordinates to the class attributes. method is complete when feed is killed.
        :param cascade: cascade to be used, object to be detected.
        :param self:
        :return:
        """

        # load cascades locally
        front_face_cascade = cv2.CascadeClassifier(self.front_face_objects)
        profile_face_cascade = cv2.CascadeClassifier(self.profile_face_objects)
        cascade = int(cascade)

        avg_pos = np.zeros((10, 4))
        counter = 0
        first_flag = 1
        window_size = 10

        while self.flag:

            ret, feed_by_frame = feed.read()
            feed_by_frame_small = cv2.resize(feed_by_frame, (0,0), fx=0.2, fy=0.2)

            # detect which object to detect

            if cascade == 0:
                self.frame_height, self.frame_width = feed_by_frame_small[:,:,0].shape
                gray_feed = cv2.cvtColor(feed_by_frame_small, cv2.COLOR_BGR2GRAY)
                flip_feed_g = cv2.flip(gray_feed, 0)
                self.find_object(gray_feed, feed_by_frame_small, front_face_cascade)
            elif cascade == 1:
                feed_by_frame_small = cv2.resize(feed_by_frame, (0,0), fx=0.2, fy=0.2)
                gray_feed = cv2.cvtColor(feed_by_frame_small, cv2.COLOR_BGR2GRAY)
                flip_feed_g = cv2.flip(gray_feed, 0)
                self.find_object(flip_feed_g, feed_by_frame_small, profile_face_cascade)
            elif cascade == 2:
                self.find_boxes(gray_feed, feed_by_frame)


            temp_pos = hw.set_servos(self.positions.copy(), self.object_coord)
            avg_pos[counter] = temp_pos 
            counter += 1
            if first_flag:
                if counter == window_size:
                    first_flag = 0
            else:
                self.position = (np.average(avg_pos,0)).tolist()
                
            counter %= window_size 


                
                
            #print(self.positions)

                # print debug
                # print("position:", self.object_coord)
                # print("velocity:", self.object_vel)
                # print("acceleration:", self.object_acc)

                # for debug, remove later
                #cv2.imshow('Frame', feed_by_frame)

    def kill_camera():
        self.kill_feed()
        return 1

    def calculate_coord(self, vertex1, vertex2, direction):
        """
        calculate_coord - gets the coordinates of the current relative position of the object in question and
        calculates the velocity and direction by using the delta between the current position and the last known
        position. Objects are stored globally and returns the current position of the object in question.
        """
        current_pos = (vertex1 + vertex2) / 2
        if direction == "x":
            prev_vel_x = self.x_vel
            old_time_x = self.curr_time_x
            if self.x_coord != 0:
                t = time.time()
                time_dif = t - old_time_x
                x_diff = current_pos - self.x_coord 
                self.curr_time_x = t
                if x_diff < 1:
                    self.x_vel = 0
                    self.x_acc = 0
                else:
                    self.x_vel = (x_diff)/time_dif
                    self.x_acc = (self.x_vel - prev_vel_x)/time_dif
                x_future = current_pos + self.x_vel * time_dif + 0.5 * self.x_acc * time_dif ** 2
                return x_future
            else:
                return current_pos
        elif direction == "y":
            prev_vel_y = self.y_vel
            old_time_y = self.curr_time_y
            if self.y_coord != 0:
                t = time.time()
                time_dif = t - old_time_y
                y_diff = current_pos - self.y_coord
                self.curr_time_y = t
                if y_diff < 1:
                    self.y_vel = 0
                    self.y_acc = 0
                else:
                    self.y_vel = (y_diff)/time_dif
                    self.y_acc = (self.y_vel - prev_vel_y)/time_dif
                y_future = current_pos + self.y_vel * time_dif + 0.5 * self.y_acc * time_dif ** 2
                return y_future
            else:
                return current_pos

        self.object_vel = [self.x_vel, self.y_vel]
        self.object_acc = [self.x_acc, self.y_acc]

    def find_object(self, feed_g, feed_color, cascade):

        front_face = cascade.detectMultiScale(feed_g, 1.1, 5)
        for (x, y, w, h) in front_face:
            cv2.rectangle(feed_color, (x, y), (x + w, y + h), (0, 255, 0), 1)

            self.x_coord = self.calculate_coord(x, x + w, "x")
            self.y_coord = self.calculate_coord(y, y + h, "y")

            normalized_x_coord = (self.x_coord/(self.frame_width/2.0)) - 1.0
            normalized_y_coord = ((self.y_coord/ (self.frame_height/2.0)) - 1.0) * (self.frame_height/self.frame_width)
            self.object_coord = [normalized_x_coord, normalized_y_coord]

    def get_coords(self):

        return self.object_coord

    def find_boxes(self, gray, img):
        """
            find_boxes - algorithm to detect and display the coordinates for objects that are like pieces of paper on
            a desk or table. Note: works specifically for these types of applications. Passes the information to the
            global variable, center_poly. can be collected from the object inference.
        """

        # find canny image of objects in frame.
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_g = cv2.bilateralFilter(img_g, 9, 75, 75)
        img_g = cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
        img_g = cv2.medianBlur(img, 11)
        img_g = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        edges = cv2.Canny(img_g, 200, 250)

        # find contours
        im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = filter(lambda cont: cv2.arcLength(cont, False) > 100, contours)
        contours = filter(lambda cont: cv2.contourArea(cont) > 10000, contours)

        # simplify contours down to polygons
        rects = []
        for cont in contours:
            rect = cv2.approxPolyDP(cont, 40, True).copy().reshape(-1, 2)
            rects.append(rect)

            # find center
            center = np.around((rect[1] / 2) + (rect[3] / 2))
            center = center.astype(int)
            self.center_poly = center
            # cv2.circle(img, tuple(center), 5, (0, 255, 0), -1)

        # draws contours to create box on image, center uses this information to calculate the object center.
        cv2.drawContours(img, rects, -1, (0, 255, 0), 1)

        # for debug.
        # print(self.center_poly)

        pass


