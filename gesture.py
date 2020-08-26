import cv2
import imutils
import numpy as np
import os
from keras.models import load_model
from keras.utils import np_utils
import time
from collections import Counter
from gtts import gTTS
import playsound



bg = None
list_np=[]
# 1- yurim(20epoch) , lr = 0.001
# 2- yurim(10epoch),lr = 0.0005
# 3- hoyeon(20 epoch)
# 4 - hoyeon2(5 epoch)
# 5 - hoyeon3(8 epoch)
#  2, 5 으로 결정
model=load_model('cnn_model_keras2')
sent = []
alpha = ["A","B","C","D","E","M","P","O","Y","LOVE","MONEY","I"]

def speak(text):
    tts = gTTS(text = text, lang = "en")
    filename = "voice.mp3"
    tts.save(filename)
    playsound.playsound(filename)

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
        blackboard = np.zeros((393, 600, 3), dtype=np.uint8)
        splitted_text = split_sentence("", 2)
        put_splitted_text_in_blackboard(blackboard, splitted_text)

        #top, right, bottom, left = 10, 350, 350, 600
        window =  cv2.rectangle(clone, (300, 50), (400, 150), (0,255,0), 2)

        # draw font
        point = right, 360
        white_color = (255, 255, 255)
        res = np.hstack((window, blackboard))
        #글자 쓰는 부분
        #cv2.putText(window, "text to speech", point, cv2.FONT_HERSHEY_SIMPLEX, 1.5 ,white_color, 2, cv2.LINE_AA)


        #directory=r'/Users/seoyulim/madcamp3/sign_language/Sign-Language/gestures/4'
        directory='/Users/seoyulim/madcamp3/sign_language/Sign-Language/input'
        #display the frame with segmented hand

        if keypress == ord("p") or (reps>0 and reps <=50):

            reps += 1
            if reps%5 == 0:
                os.chdir(directory)
                #top, right, bottom, left = 10, 350, 350, 600
                crop_img  = window[50:150,300:400]
                cv2.imwrite("test"+str(num_frames)+".jpg", crop_img)
                test = cv2.imread("test"+str(num_frames)+".jpg",0)
                test_img = np.array(test, dtype=np.uint8)
                test_img = np.resize(test_img, (100,100,1))
                list_np.append(test_img)

            if(reps >= 50) :
                reps = 0
                list_np = np.array(list_np)
                print(list_np.shape)
                y_pred = model.predict_classes(list_np)
                result = Counter(y_pred)
                label = result.most_common(n=1)
                label_id = label[0][0]
                sent.append(alpha[label_id])
                print(sent)
                #cv2.putText(blackboard, alpha[y_pred[2]], (4, 200), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
                #res = np.hstack((window, blackboard))
                list_np=[]

        sentence = ''.join(sent)
        cv2.putText(blackboard, sentence, (4, 100), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        res = np.hstack((window, blackboard))
        cv2.imshow("Video Feed", res)

        # if the user pressed "r" read text
        if keypress == ord("r"):
            speak(sentence)

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        # if the user pressed "d" delete sentence
        if keypress == ord("d"):
            sent=[]
        # if the user pressed "s" add space to sentence
        if keypress == ord("s"):
            sent.append(" ")
        # if the user pressed "b" backspace
        if keypress == ord("b"):
            del sent[-1]


# free up memory
camera.release()
cv2.destroyAllWindows()
