import cv2
import numpy as np


class Camera:
    # constructor
    x_coord = 0
    y_coord = 0
    fps = 0
    x_vel = 0
    y_vel = 0
    x_acc = 0
    y_acc = 0

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

            # self.find_object(gray_feed, feed_by_frame, front_face_cascade)
            # self.find_object(gray_feed, feed_by_frame, profile_face_cascade)

            # print("position:", self.x_coord, self.y_coord)
            # print("velocity:", self.x_vel, self.y_vel)
            # print("acceleration:", self.x_acc, self.y_acc)

            self.detect_corners(gray_feed, feed_by_frame)


            # for debug, remove later
            # cv2.imshow('Frame', feed_by_frame)
            # cv2.imshow('Frame2', gray_feed)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.kill_feed()

    def calculate_coord(self, vertex1, vertex2, direction):
        """
        calculate_coord - gets the coordinates of the current relative position of the object in question and
        calculates the velocity and direction by using the delta between the current position and the last known
        position. Objects are stored globally and returns the current position of the object in question.
        """
        current_pos = (vertex1 + vertex2) / 2
        if direction == "x":
            prev_vel_x = self.x_vel
            if self.x_coord != 0:
                self.x_vel = self.x_coord - current_pos
                self.x_acc = self.x_vel - prev_vel_x
        elif direction == "y":
            prev_vel_y = self.y_vel
            if self.y_coord != 0:
                self.y_vel = self.y_coord - current_pos
                self.y_acc = self.y_vel - prev_vel_y
        return current_pos

    def find_object(self, feed_g, feed_color, cascade):

        front_face = cascade.detectMultiScale(feed_g, 1.2, 5)
        for (x, y, w, h) in front_face:
            cv2.rectangle(feed_color, (x, y), (x + w, y + h), (0, 255, 0), 1)

            self.x_coord = self.calculate_coord(x, x + w, "x")
            self.y_coord = self.calculate_coord(y, y + h, "y")
        pass

    def get_new(old):
        new = np.ones(old.shape, np.uint8)
        cv2.bitwise_not(new, new)
        return new

    def detect_corners(self, img_gray, img):

        maxAreaFound = 0

        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_g = cv2.bilateralFilter(img_g, 9, 75, 75)
        img_g = cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
        img_g = cv2.medianBlur(img, 11)
        img_g = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        edges = cv2.Canny(img_g, 200, 250)

        im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = filter(lambda cont: cv2.arcLength(cont, False) > 100, contours)
        contours = filter(lambda cont: cv2.contourArea(cont) > 10000, contours)

        # simplify contours down to polygons
        rects = []
        for cont in contours:
            rect = cv2.approxPolyDP(cont, 40, True).copy().reshape(-1, 2)
            rects.append(rect)

        # that's basically it
        cv2.drawContours(img, rects, -1, (0, 255, 0), 1)


        cv2.imshow('canny', edges)
        cv2.imshow('canny', img)



        pass




front_face_objects = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
profile_face_objects = cv2.data.haarcascades + 'haarcascade_profileface.xml'

camera = Camera(front_face_objects, profile_face_objects, 0)
camera_ready = camera.camera_ready()
camera.detect_object()

