# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
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
from Mouth import Mouth
from Eye import Eye


class DrowsinessDetection():
        def __init__(self):
                self.cascade="/media/pi/CE4A-9378/drowsiness-detection/haarcascade_frontalface_default.xml"
                self.shape_predictor="/media/pi/CE4A-9378/drowsiness-detection/shape_predictor_68_face_landmarks.dat"
                self.ALARM_ON = False
                self.alarm="/media/pi/CE4A-9378/drowsiness-detection/alarm.wav"
                
        def start(self):
                # initialize dlib's face detector (HOG-based) and then create
                # the facial landmark predictor
                print("[INFO] loading facial landmark predictor...")
                #detector = dlib.get_frontal_face_detector()
                detector = cv2.CascadeClassifier(self.cascade)
                predictor = dlib.shape_predictor(self.shape_predictor)
                # grab the indexes of the facial landmarks for the left and
                # right eye, respectively
                (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
                (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
                (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

                # start the video stream thread
                print("[INFO] starting video stream thread...")
                #vs = VideoStream(src=args["webcam"]).start()
                vs = VideoStream(usePiCamera=True).start()
                #vs=cv2.VideoCapture("http://192.168.1.101:8080")
                time.sleep(1.0)

                '''buzzer = Buzzer(17)
                buzzer.on();
                '''
                GPIO.setwarnings(False) 
                #Select GPIO mode 
                GPIO.setmode(GPIO.BCM) 
                #Set buzzer - pin 23 as output 
                buzzer=17
                GPIO.setup(buzzer,GPIO.OUT) 

                mouthobj=Mouth();
                lefteyeobj=Eye();
                righteyeobj=Eye();

                #url="http://192.168.1.101:8080/shot.jpg"
                # loop over frames from the video stream
                while True:
                        # grab the frame from the threaded video file stream, resize
                        # it, and convert it to grayscale
                        # channels)
                        #imgResp=urllib.request.urlopen(url)
                        #imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
                        #img=cv2.imdecode(imgNp,-1)
                        #cv2.imshow('test',img)
                        frame = vs.read()
                        frame = imutils.resize(frame, width=450)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # detect faces in the grayscale frame
                        #rects = detector(gray, 0)

                        rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
                                minNeighbors=5, minSize=(30, 30),
                                flags=cv2.CASCADE_SCALE_IMAGE)

                        # loop over the face detections
                        for (x, y, w, h) in rects:
                                rect = dlib.rectangle(int(x), int(y), int(x + w),
                                        int(y + h))
                 
                                # determine the facial landmarks for the face region, then
                                # convert the facial landmark (x, y)-coordinates to a NumPy
                                # array
                                shape = predictor(gray, rect)
                                shape = face_utils.shape_to_np(shape)
                       
                                # extract the left and right eye coordinates, then use the
                                # coordinates to compute the eye aspect ratio for both eyes
                                lefteyeobj.eye = shape[lStart:lEnd]
                                righteyeobj.eye = shape[rStart:rEnd]
                                mouthobj.mouth=shape[mStart:mEnd]
                                #print mouth
                                #[48:68]
                                lefteyeobj.eyeEAR = lefteyeobj.eye_aspect_ratio()
                                righteyeobj.eyeEAR = righteyeobj.eye_aspect_ratio()
                                mouthobj.mouthEAR = mouthobj.mouth_aspect_ratio()

                                # average the eye aspect ratio together for both eyes
                                ear = (lefteyeobj.eyeEAR + righteyeobj.eyeEAR) / 2.0

                                # compute the convex hull for the left and right eye, then
                                # visualize each of the eyes
                                lefteyeobj.eyeHull = cv2.convexHull(lefteyeobj.eye)
                                righteyeobj.eyeHull = cv2.convexHull(righteyeobj.eye)
                                mouthobj.mouthHull = cv2.convexHull(mouthobj.mouth)
                                cv2.drawContours(frame, [lefteyeobj.eyeHull], -1, (0, 255, 0), 1)
                                cv2.drawContours(frame, [righteyeobj.eyeHull], -1, (0, 255, 0), 1)
                                cv2.drawContours(frame, [mouthobj.mouthHull], -1, (0, 255, 0), 1)

                                # check to see if the eye aspect ratio is below the blink
                                # threshold, and if so, increment the blink frame counter
                                if ear < lefteyeobj.EYE_AR_THRESH:
                                        lefteyeobj.COUNTER_EYE += 1
                                        print("Eye{}".format(lefteyeobj.COUNTER_EYE))
                                        if lefteyeobj.COUNTER_EYE >= lefteyeobj.EYE_AR_CONSEC_FRAMES:
                                                # if the alarm is not on, turn it on
                                                if not self.ALARM_ON:
                                                        self.ALARM_ON = True

                                                        # check to see if an alarm file was supplied,
                                                        # and if so, start a thread to have the alarm
                                                        # sound played in the background
                                                        '''if args["alarm"] != "":
                                                                t = Thread(target=sound_alarm,
                                                                        self.alarm)
                                                                t.deamon = True
                                                                t.start()'''
                                                        #elif args["alarm"] == "":
                                                        count=0
                                                        while count<=0:
                                                                if(count<=4):
                                                                        GPIO.output(buzzer,GPIO.HIGH) 
                                                                print ("Beep") 
                                                                sleep(0.5) # Delay in seconds 
                                                                GPIO.output(buzzer,GPIO.LOW) 
                                                                print ("No Beep") 
                                                                sleep(0.5)
                                                                count+=1

                                                                
                                                                

                                                # draw an alarm on the frame
                                                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                else:
                                        #mouthobj.COUNTER_MOUTH=0
                                        print("Else in eye");
                                        lefteyeobj.COUNTER_EYE = 0
                                        self.ALARM_ON = False

                                if mouthobj.mouthEAR > mouthobj.MOUTH_AR_THRESH:
                                        mouthobj.COUNTER_MOUTH += 1
                                        print("MouthCounter"+str(mouthobj.COUNTER_MOUTH))
                                        # if the eyes were closed for a sufficient number of
                                        # then sound the alarm
                                        if mouthobj.COUNTER_MOUTH >= mouthobj.MOUTH_AR_CONSEC_FRAMES:
                                                # if the alarm is not on, turn it on
                                                if not self.ALARM_ON:
                                                        self.ALARM_ON = True

                                                        # check to see if an alarm file was supplied,
                                                        # and if so, start a thread to have the alarm
                                                        # sound played in the background
                                                        '''if args["alarm"] != "":
                                                                t = Thread(target=sound_alarm,
                                                                        self.alarm)
                                                                t.deamon = True
                                                                t.start()
                                                        elif args["alarm"] == "":'''
                                                        count=0
                                                        while count<=0:
                                                                if(count<=4):
                                                                        GPIO.output(buzzer,GPIO.HIGH) 
                                                                print ("Beep") 
                                                                sleep(0.5) # Delay in seconds 
                                                                GPIO.output(buzzer,GPIO.LOW) 
                                                                print ("No Beep") 
                                                                sleep(0.5)
                                                                count+=1


                                                # draw an alarm on the frame
                                                cv2.putText(frame, "YAWNING ALERT!", (10, 30),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                                # otherwise, the eye aspect ratio is not below the blink
                                # threshold, so reset the counter and alarm
                                else:
                                        print("Else in mouth");
                                        mouthobj.COUNTER_MOUTH=0
                                        #lefteyeobj.COUNTER_EYE=0
                                        self.ALARM_ON = False

                                # draw the computed eye aspect ratio on the frame to help
                                # with debugging and setting the correct eye aspect ratio
                                # thresholds and frame counters
                                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                 
                        # show the frame
                        cv2.imshow("Frame", frame)
                        key = cv2.waitKey(1) & 0xFF
                 
                        # if the `q` key was pressed, break from the loop
                        if key == ord("q"):
                                break

                # do a bit of cleanup
                cv2.destroyAllWindows()
                vs.stop()



                
        

        def sound_alarm(path):
                # play an alarm sound
                playsound.playsound(path)


DrowsinessDetection().start()
''' 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
        help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
        help="index of webcam on system")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.3#0.27
EYE_AR_CONSEC_FRAMES = 15 #25 #48

MOUTH_AR_THRESH = 0.1
MOUTH_AR_CONSEC_FRAMES = 10 #48


# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER_EYE = 0
ALARM_ON = False

COUNTER_MOUTH = 0

'''
