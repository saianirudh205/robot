import cv2
import time
import loading as ld
import numpy as np
from PIL import Image
import time
from resizeimage import resizeimage
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import pyttsx3
engine = pyttsx3.init()

def cap():
   video=cv2.VideoCapture(0)
   a=1
   while True:
      a=a+1
      check,frame=video.read()
   #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      cv2.imshow('capturing',frame)
      if a%1000==0:
         dct=getImage(frame)
         if dct[0] >=0 and dct[0]<1:
            if dct[0] > 0.5:
               engine.say('Hello Ms.Samantha.')
               engine.runAndWait()
               break
            else:
               engine.say('Hello Mr.Sai Anirudh.')
               engine.runAndWait()
               break
         else:
            continue
         
            
      key=cv2.waitKey(1)
      if key==ord('q'):
         break
   print(a)
   video.release()
   cv2.destroyAllWindows()



def getImage(img):
   #img = cv2.imread(name)
   face = cv2.CascadeClassifier("anirudh.xml")
   gryimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   gary = face.detectMultiScale(gryimg,scaleFactor=1.05,minNeighbors=5)
   for x,y,w,h in gary :
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,150),3)
        img = img[y:y+h, x:x+w]
   crop_img=Image.fromarray(img)
   cover = resizeimage.resize_cover(crop_img, [28, 28],validate=False)
   return ld.predict(cover)



   
'''
check,frame=video.read()
time.sleep(3)

cv2.imshow('capturni',frame)

cv2.waitKey(0)
'''
