import sys
import cv2


class Camera:
    # constructor
    x_coord = 0
    y_coord = 0

    def __init__(self, object_cascade, camera_port):
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

    def camera_ready(self):
        """
        camera_ready "turns on the camera, returns a true or false value for the camera open function, 0 is default for a
        built-in laptop webcam, will change if different for rbPI"
        :param self:
        :param feed: camera object
        :return: boolean representation of camera on.
        """
        return self.feed.isOpened()

    def detect_object(self):
        """
        detect object - method detects an object given the specified object perameters given by the haar cascade and sends
        x and y coordinates to the class attributes. method is complete when feed is killed.
        :param self:
        :return:
        """
        object_cascade = cv2.CascadeClassifier(self.object_cascade)

        while self.feed:
            ret, feed_by_frame = self.feed.read()
            gray_feed = cv2.cvtColor(feed_by_frame, cv2.COLOR_BGR2GRAY)

            objects = object_cascade.detectMultiScale(gray_feed, 1.5, 5)
            for (x, y, w, h) in objects:
                cv2.rectangle(feed_by_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.x_coord = (x + (x + w)) / 2
                self.y_coord = (y + (y + h)) / 2
                # print(x_coord, y_coord)

            cv2.imshow('Frame', feed_by_frame)
            cv2.imshow('Frame2', gray_feed)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.feed.release()
        cv2.destroyAllWindows()

    def kill_feed(self):
        # When everything done, release the video capture object
        self.feed.release()

        # Closes all the frames
        cv2.destroyAllWindows()


face_objects = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

camera = Camera(face_objects, 0)
camera_ready = camera.camera_ready()
camera.detect_object()


# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# cap = cv2.VideoCapture(0)
#
# # Check if camera opened successfully
# if not cap.isOpened():
#     print("Error opening video stream or file")
#
# # Read until video is completed
# while cap.isOpened():
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if ret:
#
#         # Display the resulting frame
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         # face detector
#         faces = face_cascade.detectMultiScale(gray, 1.1, 5)
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             x_coord = (x + (x + w)) / 2
#             y_coord = (y + (y + h)) / 2
#             # print(x_coord, y_coord)
#
#         cv2.imshow('Frame', frame)
#         cv2.imshow('Frame2', gray)
#
#         # Press Q on keyboard to  exit
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#
#     # Break the loop
#     else:
#         break
#
# # When everything done, release the video capture object
# cap.release()
#
# # Closes all the frames
# cv2.destroyAllWindows()
