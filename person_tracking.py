import numpy as np
import cv2
import time
import imutils
import datetime
import argparse
#import spectral as spc
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
#pts = deque(maxlen=args["buffer"])

person_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
# face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)
 
# otherwise, we are reading from a video file
else:
#	time.sleep(0.10)
	camera = cv2.VideoCapture(args["video"])

firstFrame = None

xpoints = np.array([[]], np.int32)

ypoints = np.array([[]], np.int32)
numObjects = 0

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	time.sleep(0.03)
	
	peopleCount = 0
	carCount = 0
	
	(grabbed, frame) = camera.read()
	# frame = frame[0:500, 0:500]
	text = "Unoccupied"
 
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break

	# Limit based on number of objects
	blurLimit = 0
#	if (numObjects>20):
#		blurLimit = 18
#	elif (numObjects>14):
#		blurLimit = 6
#	else:
#		blurLimit = 0
 
	# resize the frame, convert it to grayscale, and blur it
	frame = cv2.resize(frame, (500,375))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5 + blurLimit, 5 + blurLimit), 0)

 	#print frame.shape
	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 	# firstFrame = gray
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	_, cnts, _= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	numObjects = len(cnts)
	# # (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	# 	cv2.CHAIN_APPROX_SIMPLE)
	index = 0

	# loop over the contours
	for c in cnts:
		if (len(cnts)>20):
			limit = 350
		elif (len(cnts)>16):
			limit = 90
		else:
			limit = 15

		# if the contour is too small, ignore it
		if cv2.contourArea(c) < limit:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text

		(x, y, w, h) = cv2.boundingRect(c)


		# Checks for height and width ratio to detect people
		if ((w*8)/5 <= h) or (cv2.contourArea(c) < 50):
			cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0, 0), 2)
			peopleCount += 1
		else:
			carCount += 1
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		xpoints = np.append(xpoints, x)
		ypoints = np.append(ypoints, y)

		for coord in xrange(0,len(xpoints)):
			if (x != xpoints[coord] or y != ypoints[coord]):
		 		if ((w*8)/5 <= h) or (cv2.contourArea(c) < 50):
		                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0, 0), 2)
        		                #peopleCount += 1
                		else:
                	        	#carCount += 1
                        		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


		if (index == 20):
		 	index = 0
		 	xpoints = np.array([[]], np.int32)
		 	ypoints = np.array([[]], np.int32)

		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		people = person_cascade.detectMultiScale(roi_gray)
		text = "Occupied"


		# identify the object
		# cv2.putText(frame, "Object".format(text), (x,y),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
		# for (ex,ey,ew,eh) in people:
		# 	cv2.putText(frame, (xpoints[i], ypoints[i]), 1, (0, 255, 0), thickness=1, lineType=8, shift=0) 
		index+=1
	# draw the text and timestamp on the frame

	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	
	# Display people and car count
	cv2.putText(frame, "People: {}".format(peopleCount), (250, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 2)
	cv2.putText(frame, "Objects: {}".format(carCount), (350, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


	pts1 =	np.float32([(170,56),(284,87),(169,173),(286,183)])
	pts2 = np.float32([(0,337),(0,291),(0,337),(0,291)])  
	# show the frame and record if the user presses a key
	# hostage = frame[0:200, 0:300]
	cv2.imshow("webcam feed", frame)
	# cv2.imshow("webcam feed hostage view", hostage)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)

	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key is pressed, break from the loop
	if key == ord("q"):
		break
# cleanup the camera and close any open windows
userAsk = "none"
while str(userAsk) != "3":
	userAsk = input("To get more info about cars type 1, to get more infor about people type 2, to close the program type 3 \n")
	if str(userAsk) == "1":
		print "Fetching details about cars..."
	elif str(userAsk) == "2":
		print "Fetching details about people..."
	elif str(userAsk) == "3":
		print "Exiting..."
	else:
		print "This is not a possible command"
	#print userAsk
camera.release()
cv2.destroyAllWindows()
