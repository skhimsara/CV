import cv2
import sys

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier("/Users/sidnpoo/Downloads/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("/Users/sidnpoo/Downloads/opencv-3.1.0/data/haarcascades/haarcascade_eye.xml")
smilePath = "/Users/sidnpoo/Downloads/opencv-3.1.0/data/haarcascades/haarcascade_smile.xml"
smileCascade = cv2.CascadeClassifier(smilePath)


video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=20,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

        # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]  #part of face inside the rectangle
        eyes = eye_cascade.detectMultiScale(roi_gray,
                                                scaleFactor=1.1,
                                                minNeighbors=20,
                                                minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE) #look only in that rectangle
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)

	smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.1,
            minNeighbors=22,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
            )

        # Set region of interest for smiles
	for (x, y, w, h) in smile:
            print "Found", len(smile), "smiles!"
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (0, 255, 0), 1)
            #print "!!!!!!!!!!!!!!!!!"


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
