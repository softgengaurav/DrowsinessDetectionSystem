from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO 
from gpiozero import Buzzer
from time import sleep
import numpy as np
import playsound
import urllib.request
import argparse
import imutils
import time
import dlib
import cv2

class Eye():
    def __init__(self):
        self.eye=0
        self.eyeEAR=0
        self.eyeHull=0
        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 15 #25 #48
        self.COUNTER_EYE=0

    def eye_aspect_ratio(self):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(self.eye[1], self.eye[5])

        B = dist.euclidean(self.eye[2], self.eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(self.eye[0], self.eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear

        
