# Python program to illustrate HoughLine 
# method for line detection 
import cv2 
import numpy as np 
import keyboard
import time

def lowers(blue1, green1, red1):
	ret = []
	if keyboard.is_pressed('r'):
		if(red1>0):
			red1-=1
	if keyboard.is_pressed('g'):
		if(green1>0):
			green1-=1
	if keyboard.is_pressed('b'):
		if(blue1>0):
			blue1-=1
	
	ret.append(blue1)
	ret.append(green1)
	ret.append(red1)

	return ret
	
def raises(blue1, green1, red1):
	ret = []
	if keyboard.is_pressed('r'):
		if(red1<255):
			red1+=1
	if keyboard.is_pressed('g'):
		if(green1<255):
			green1+=1
	if keyboard.is_pressed('b'):
		if(blue1<255):
			blue1+=1

	ret.append(blue1)
	ret.append(green1)
	ret.append(red1)

	return ret

def log(image,clr,color,off):
	font = cv2.FONT_HERSHEY_SIMPLEX
	org = (5, 30+off)
	fontScale = 1
#	color = (255, 0, 0)
	thickness = 2
	image = cv2.putText(image, str(clr), org, font, 
					fontScale, color, thickness, cv2.LINE_AA)
 
cap = cv2.VideoCapture(0)
f = False
t = False
rgb1 = [0,0,0]
rgb2 = [255,255,255]

while(cap.isOpened()):
	ret, image = cap.read()
	
	boundaries = [
		(rgb1, rgb2)]

	for (lower, upper) in boundaries:
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
		# find the colors within the specified boundaries and apply
		# the mask
		mask = cv2.inRange(image, lower, upper)
		img= cv2.bitwise_and(image, image, mask = mask)
		
		# Convert the img to grayscale 
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
	#	cv2.imwrite('gray.jpg', gray)
		  
		# Apply edge detection method on the image 
	#	edges = cv2.Canny(gray,50,150,apertureSize = 3) 
	#	cv2.imwrite('edges.jpg', edges)
		  
		# This returns an array of r and theta values 
	#	lines = cv2.HoughLines(edges,1,np.pi/180, 360) 
		
		
	#	ret, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
		contours, hierarchy = cv2.findContours(gray, 1, 2)  

		if keyboard.is_pressed('f'):
			f = not f
			time.sleep(0.1)
		if f:
			image = img
		
		if keyboard.is_pressed('t'):
			t = not t
			time.sleep(0.05)
		
		if keyboard.is_pressed('d'):
			rgb1 = [0,0,0]
			rgb2 = [255,255,255]

    
		if(len(contours) > 0): 
			cMax = max(contours, key=cv2.contourArea)
			rect = cv2.minAreaRect(cMax)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			cv2.drawContours(image, [box], cv2.FILLED, (255,0,127), 3)
			
			circleRect = cv2.boundingRect(cMax)
			x,y,w,h = circleRect
			frect = (x+(int)(w/4),y+(int)(h/4),(int)(w/2),(int)(h/2))
			x,y,w,h = frect
			#cv2.circle(image,(int(x+(w/2)), int(y+(h/2))), 6, (0,255,0), 2)
			cv2.rectangle(image,frect,(0,255,0), 2)

		if keyboard.is_pressed('down'):
			if(t):
				rgb2 = lowers(rgb2[0],rgb2[1],rgb2[2])
			else:
				rgb1 = lowers(rgb1[0],rgb1[1],rgb1[2])
		
		if keyboard.is_pressed('up'):
			if(t):
				rgb2 = raises(rgb2[0],rgb2[1],rgb2[2])
			else:
				rgb1 = raises(rgb1[0],rgb1[1],rgb1[2])

		avg = np.array(cv2.mean(image[y:y+h,x:x+w])).astype(np.uint8)
		rgb = avg.tolist()		
		if keyboard.is_pressed('l'):

			i = 2
			while(i>=0):
				rgb1[i]=rgb[i]-50
				rgb2[i]=rgb[i]+50
				if rgb1[i] < 0 :
					rgb1[i]=0
				if rgb2[i] > 255 :
					rgb2[i] = 255
				i-=1
				time.sleep(0.07)
			print('Locked on to: ',np.array(cv2.mean(image[y:y+h,x:x+w])).astype(np.uint8))



		boundaries = [
		(rgb1, rgb2)]

		log(image,rgb1[0],[255,0,0],0)	
		log(image,rgb1[1],[0,255,0],30)
		log(image,rgb1[2],[0,0,255],60)	

		log(image,rgb2[0],[255,0,0],120)	
		log(image,rgb2[1],[0,255,0],150)
		log(image,rgb2[2],[0,0,255],180)
		log(image,'fltr' if f else 'cam',[255,0,255],240)	
		log(image,'hiB' if t else 'loB',[255,0,255],270)
		log(image,rgb,rgb,400) #does not function well
		cv2.imshow('frame',image)
		cv2.waitKey(40)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
       
            

cap.release()
cv2.destroyAllWindows()
