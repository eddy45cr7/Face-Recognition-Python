import cv2

faceCascade = cv2.cascadeClassifier('haarcascade_frontalface_default.xml')
video_capture=cv2.VideoCapture(3)

while True:
    #capture frame by frame
    ret ,frame = video_capture.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY())
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.cv.CV_HARR_SCALE_IMAGE

    )
    # Draw a rectangle around a face
    for(x,y,w,h) in faces :
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
       
        #display the resulting frame
    cv2.imshow('video',frame)
    if cv2.waitkey(1) & 0xff== ord('q'):
        break

        #when everything is done , release the capture
video_capture.release() 
cv2.destroyALLWindow()    