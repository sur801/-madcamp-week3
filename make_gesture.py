import cv2
import imutils
import numpy as np
import os
from keras.models import load_model
from keras.utils import np_utils
import time
from collections import Counter


bg = None
list_np=[]


#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def split_sentence(text, num_of_words):
	'''
	Splits a text into group of num_of_words
	'''
	list_words = text.split(" ")
	length = len(list_words)
	splitted_sentence = []
	b_index = 0
	e_index = num_of_words
	while length > 0:
		part = ""
		for word in list_words[b_index:e_index]:
			part = part + " " + word
		splitted_sentence.append(part)
		b_index += num_of_words
		e_index += num_of_words
		length -= num_of_words
	return splitted_sentence

def put_splitted_text_in_blackboard(blackboard, splitted_text):
	y = 200
	for text in splitted_text:
		cv2.putText(blackboard, text, (4, y), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		y += 50

#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 350, 600
    #top, right, bottom, left = 50, 400, 100, 300
    # initialize num of frames
    num_frames = 0
    reps = 0

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)


        # increment the number of frames
        num_frames += 1

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # draw black blackboard
        blackboard = np.zeros((393, 393, 3), dtype=np.uint8)
        splitted_text = split_sentence("", 2)
        put_splitted_text_in_blackboard(blackboard, splitted_text)

        #top, right, bottom, left = 10, 350, 350, 600
        window =  cv2.rectangle(clone, (300, 50), (400, 150), (0,255,0), 2)

        # draw font
        point = right, 360
        white_color = (255, 255, 255)

        directory='/Users/seoyulim/madcamp3/sign_language/Sign-Language/gestures/11'
        #display the frame with segmented hand

        if keypress == ord("p") or 0<reps<2000:
            reps += 1
            os.chdir(directory)
            #top, right, bottom, left = 10, 350, 350, 600
            crop_img  = window[50:150,300:400]
            cv2.imwrite(str(reps+6000)+".jpg", crop_img)
            print("save"+str(reps))
            if(reps==2000):
                print("#####done######")
        cv2.imshow("video feed",window)
        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
# free up memory
camera.release()
cv2.destroyAllWindows()
