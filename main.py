import cv2
import time
import glob
import os
from emailing import send_email
from threading import Thread

#set camera
video = cv2.VideoCapture(0)
#give time for camera to prepare
time.sleep(1)

first_frame = None
status_list = []
count = 1

#clean images folder:
def clean_folder():
    print("clean folder function started")
    images = glob.glob("images/*.png")
    for image in images:
        os.remove(image)
    print("clean folder function ended")

# Define clean_thread outside the loop
clean_thread = Thread(target= clean_folder)
clean_thread.daemon = True

while True:
    status = 0
    #webcam only print read in terminal, not shown
    check, frame = video.read()
    #convert to gray frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #reduce noise with gaussian blur
    gray_frame_gau = cv2.GaussianBlur(gray_frame,(21,21),0)

    #set first frame
    if first_frame is None:
        first_frame = gray_frame_gau

    #differentiate each frame with first frame
    delta_frame = cv2.absdiff(first_frame, gray_frame_gau)

    #reduce noise
    thresh_frame = cv2.threshold(delta_frame, 60, 255,cv2.THRESH_BINARY)[1]
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)

    #webcam show
    #cv2.imshow("My video", dil_frame)

    #diff real object and fake object
    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #create rectangle around real object
    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rectangle = cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)

        #send email when rectangle detect objects
        if rectangle.any():
            status = 1
            #extract image from webcam
            cv2.imwrite(f"images/{count}.png", frame)
            count = count + 1

            #select 1 image (the middle one)
            #all_images = glob.glob("images/*.png")
            #index = int(len(all_images)/2)
            #image_with_object = all_images[index]

    #Initialize with a default value
    image_with_object = None

     #object enter the frame, then exit, then send email, and clean images folder
    status_list.append(status)
    status_list = status_list[-2:]

    if status_list[0] == 1 and status_list[1] == 0:

        # select 1 image (the middle one)
        all_images = sorted(glob.glob("images/*.png"), key=os.path.getctime)
        index = int(len(all_images) / 2)
        image_with_object = all_images[index]

        email_thread = Thread(target=send_email, args=(image_with_object, ))
        email_thread.daemon = True
        #clean_thread = Thread(target= clean_folder)
        #clean_thread.daemon = True

        email_thread.start()

        # Wait for email_thread to finish before starting clean_thread
        email_thread.join()

        # Start clean_thread after email_thread has finished
        clean_thread.start()

    print(status_list)

    # webcam show
    cv2.imshow("Video", frame)
    #close webcam
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

video.release()

clean_thread.join()