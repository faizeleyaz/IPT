https://codeshare.io/adwO8K
https://drive.google.com/drive/folders/1VRzVM7uJ94rz0_kf6Y97ISMAl__0oE6k?usp=sharing
run this code in jupiter notebooks


import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

image=cv2.imread('group_img.jpg')
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image1=image.copy()
plt.imshow(image)

!pip install mtcnn

from mtcnn.mtcnn import MTCNN

face_detector=MTCNN()
results=face_detector.detect_faces(image)

len(results)

#extract only face regions
faces=[]
for i in range(len(results)):
    # extract the bounding box from the first face
    x1, y1, width, height = results[i]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    cv2.rectangle(image1, (x1, y1), (x2, y2), (255,0,0), 10)
    # extract the face
    face = image[y1:y2, x1:x2]
    faces.append(face)

plt.imshow(image1)

for i in range(len(faces)):
    plt.imshow(faces[i])
    plt.show()
    
    
#Face detection

import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('aman.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0)
while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 4), 10)

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()


#Face and eye detector

import cv2 
import numpy as np
# HAAR Cascade files
cascade_face = 'haarcascade_frontalface_default.xml'
cascade_eye = 'haarcascade_eye.xml'

face_classifier = cv2.CascadeClassifier(cascade_face)
eye_classifier = cv2.CascadeClassifier(cascade_eye)

def face_and_eye_detector(image):

# Reading images and converting to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detecting faces
	faces = face_classifier.detectMultiScale(gray, 1.2, 5)
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,0), 3)

		# Cropping the face found
		area_gray = gray[y:y+h, x:x+w]
		area_original = image[y:y+h, x:x+w]

		# Detecting eyes
		eyes = eye_classifier.detectMultiScale(area_gray)
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(area_original, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
		
	image = cv2.flip(image, 1)		
	return image		

	if faces is ():
		return image		


capture = cv2.VideoCapture(0)

while True:
	response, frame = capture.read()
	cv2.imshow("Live Face and Eye Classifier", face_and_eye_detector(frame))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

capture.release()
cv2.destroyAllWindows()

#Car detection
import numpy as np
import cv2
import time

car_cascade = 'haarcascade_car.xml'
car_classifier = cv2.CascadeClassifier(car_cascade)
capture = cv2.VideoCapture('cars.avi')

while capture.isOpened():

    response, frame = capture.read()
    if response:

    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    	cars = car_classifier.detectMultiScale(gray, 1.2, 3) 

    	for (x, y, w, h) in cars:
    		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,0), 3)
    		cv2.imshow('Cars', frame)

    	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break
    else:
    	break    	

        	
capture.release()
cv2.destroyAllWindows() 

#Road Lane detector
import cv2
import numpy as np

video =cv2.VideoCapture("road_car_view.mp4")

while True:
    ret,or_frame=video.read()
    if not ret :
        video =cv2.VideoCapture("road_car_view.mp4")
        continue
    frame=cv2.GaussianBlur(or_frame,(5,5),0)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    lower_y=np.array([18,94,140])
    upper_y=np.array([48,255,255])
    
    mask=cv2.inRange(hsv,lower_y,upper_y)
    edges=cv2.Canny(mask,74,150)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,50,maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),5)
    
    cv2.imshow("frame",frame)
    cv2.imshow("edges",edges)
    key=cv2.waitKey(25)
    if(key==27):
        break
    


video.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
