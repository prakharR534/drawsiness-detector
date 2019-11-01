#importing necessory packets

import cv2
import dlib
import time
import imutils
import argparse
import playsound
import numpy as np
from threading import Thread 
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance as dist 


def sound_alarm(path):
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	C = dist.euclidean(eye[0], eye[3])

	ear = (A+B) / (2.0 * C)

	return ear

ap = argparse.ArgumentParser()
ap.add_argument("-p","--shape-predictor", required=True,help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,	help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
#If the eye aspect ratio falls below this threshold, we’ll start counting the number of frames the person has closed their eyes for.

C = 0
ALARN_ON = False


print("loading facial landmark predictor......")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape-predictor(args["shape-predictor"])






(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]