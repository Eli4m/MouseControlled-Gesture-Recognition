import cv2
import numpy as np
from pynput.mouse import Button, Controller
import wx

#screen size of monitor
app = wx.App(False)
#screen x and y coordinates 
(sx,sy) = wx.GetDisplaySize()
(camx,camy) = (320,240)

cam = cv2.VideoCapture(0)
cam.set(3,camx)
cam.set(4,camy)

mouse = Controller()

kernalOpen = np.ones((7,7))
kernalClosed = np.ones((15,15))

#previous mouse coordinate
pMLoc = np.array([0,0])
#mouse coord after dampning
mLoc = np.array([0,0])
Damping = 2

closed = False
while True:
    ret, img = cam.read()
    img = cv2.flip(img,1)
    #Resize for better prcessing 
    img = cv2.resize(img,(340,220))

    #convert BGR to HSV
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #SKIN COLOR MASK
    #lower_skin = np.array([0,20,70],dtype = np.uint8)
    #upper_skin = np.array([20,255,255],dtype = np.uint8)
    #mask = cv2.inRange(hsv,lower_skin,upper_skin)
    
    #BLUE MASK
    lower_blue = np.array([100,150,0])
    upper_blue = np.array([140,255,255])
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    
    #Morphology
    #if kernals is smaller leave the cluster of pixels where it is
    maskOpen = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernalOpen)
    maskClosed = cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernalClosed)

    maskFinal = maskClosed
    conts,h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    #check if open gesture or closed gesture for 2 fingers
    ##if conts = 2 then its an opend gesture
    if(len(conts) == 2):
        if(closed == False):
            closed = True
            mouse.release(Button.left)
        x1,y1,w1,h1 = cv2.boundingRect(conts[0])
        x2,y2,w2,h2 = cv2.boundingRect(conts[1])
        #draw rectangle around fingers
        cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(0,255,0),1)
        cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(0,255,0),1)
        #draw a line from center of both fingers
        ##cx = center width, cy = center height
        cx1 = int(x1+w1/2)
        cy1= int(y1+h1/2)
        cx2 = int(x2+w2/2)
        cy2 = int(y2+h2/2)
        cx = int((cx1 + cx2)/2)
        cy = int((cy1 + cy2)/2)
        cv2.line(img,(cx1,cy1),(cx2,cy2),(255,0,0),1)
        cv2.circle(img,(cx,cy),1,(0,0,255),2)
        #damp mouse Pos
        mLoc = pMLoc+((cx,cy)-pMLoc)/Damping
        #control mouse position with cirple pos
        #convert camx and y to x and y
        mouseLoc = int((mLoc[0]*sx/camx)),int((mLoc[1]*sy/camy))
        mouse.position = mouseLoc
        #wait untill mouse is in this position
        while mouse.position != mouseLoc:
            pass
        pMLoc = mLoc
        ##Bounding box for both fingers
        openX, openY, openW, openH = cv2.boundingRect(np.array([[[x1,y1],[x1+w1,y1+h1],[x2,y2],[x2+w2,y2+h2]]]))
        cv2.rectangle(img,(openX,openY),(openX+openW,openY+openH),(255,0,0),1)
    #closed gesture
    elif(len(conts) == 1):
        x,y,w,h = cv2.boundingRect(conts[0])
        if(closed == True):
            #detect if fingers are out of screen calculate size of box
            #if its smaller than 2 closed fingers its not closed
            if(abs((w*h-openW*openH)*100/(openW*openH))<30):
                closed = False
                print("clicking")
                mouse.press(Button.left)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
            #center of bounding box
            cx = int(x + w /2)
            cy = int(y + h /2)
            cv2.circle(img,(cx,cy),int((w+h)/8),(0,0,255),1)
            mLoc = pMLoc+((cx,cy)-pMLoc)/Damping
            mouseLoc = int((mLoc[0]*sx/camx)),int((mLoc[1]*sy/camy))
            mouse.position = mouseLoc
            while mouse.position != mouseLoc:
                pass
            pMLoc = mLoc
    cv2.imshow('mask',mask)
    ##masks for debugging
    cv2.imshow('openKernal',maskOpen)
    cv2.imshow('closedKernal',maskClosed)
    cv2.imshow('camNormal',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
