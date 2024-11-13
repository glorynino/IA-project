import cv2
import mediapipe as mp
import time

camera = cv2.VideoCapture(1)
ptime = 0

mpdraw = mp.solutions.drawing_utils
mpfacemesh = mp.solutions.face_mesh


facemesh = mpfacemesh.facemesh(maxface = 2)
drawspec = mpdraw.Drawingspec(thickness = 2 , circle_radius =2)

