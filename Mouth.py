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

class Mouth():
    def __init__(self):
        self.mouth=0
        self.mouthEAR=0
        self.mouthHull=0
        self.MOUTH_AR_THRESH = 0.1
        self.MOUTH_AR_CONSEC_FRAMES = 10
        self.COUNTER_MOUTH=0

    def mouth_aspect_ratio(self):#49-68 indexes 48-67
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(self.mouth[13], self.mouth[19])
        B = dist.euclidean(self.mouth[15], self.mouth[17])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(self.mouth[0],self.mouth[6])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear
        
