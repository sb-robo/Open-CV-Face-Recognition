import cv2
import numpy as np

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip=0
face_data=[]
datapath='./data/'
filename=input("Enter the Name : ")

while True:
	ret,frame=cap.read()

	if ret==False:
		continue

	faces=face_cascade.detectMultiScale(frame,1.3,5)
	if len(faces)==0:
		continue
	

	faces=sorted(faces,key=lambda f:f[2]*f[3])
	#print(faces)

	for face in faces[-1:]:
		x,y,w,h=face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)


	# Extract the croped section
		offset=10
		face_sec=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_sec=cv2.resize(face_sec,(100,100))

		skip +=1
		if (skip%10)==0:
			face_data.append(face_sec)
			print(len(face_data))

	cv2.imshow('Video frame',frame)
	cv2.imshow('Extract',face_sec)

	if (cv2.waitKey(1) & 0xFF)==ord('q'):
		break

# Convert face list in np array
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))

#save the nparray in file
np.save(datapath+filename+'.npy',face_data)
print("File Successfully saved at "+datapath+filename+'.npy')


cap.release()
cv2.destroyAllWindows()