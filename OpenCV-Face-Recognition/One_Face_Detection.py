# Load  the required trained XML classifier
import cv2
# Capture frames from a camera or image
detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
imp_img = cv2.VideoCapture("elon.jpg")

res, img = imp_img.read()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Dectects faces of different size in the input Image
# (gray_image, Scale Factor.minneighbor)
faces = detect.detectMultiScale(gray, 1.3, 5)

#(image , pt1,pt2,color,thikness)
# pt1- vertex of rectangle
# pt2- Vectex of rectangle opp. to pt1
# Color in hex format

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

cv2.imshow("Elon Image", img)

cv2.waitKey(0)
imp_img.release()
cv2.destroyAllWindow()
