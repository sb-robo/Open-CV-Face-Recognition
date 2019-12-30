import numpy as np 
import cv2
import os

# knn code

def euclideanDist(X1,X2):
    return np.sqrt(sum((X1-X2)**2))

def knn(X1,Y1,querypoint,k=11):
    distance=[]
    m=X1.shape[0]
    
    for i in range(m):
        dist=euclideanDist(querypoint,X1[i])
        distance.append((dist,Y1[i]))
        
    distance=sorted(distance)
    distance=distance[:k]
    
    distance=np.array(distance)
    new_dist=np.unique(distance[:,1],return_counts=True)
    
    index=new_dist[1].argmax()
    pred=new_dist[0][index]
    
    return pred

# camera init
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

# face detect
face_cascade=cv2.CascadeClassifier('Haarcascade_frontalface_alt.xml')

skip=0
data_path='./data/'

face_data=[]
labels=[]

class_id=0
names={} #map between id-name


# data manipulation

for fx in os.listdir(data_path):
    if fx.endswith('.npy'):
        names[class_id]=fx[:-4]
        print('Loaded '+fx)
        data_item=np.load(data_path+fx)
        face_data.append(data_item)

        # create labels for the class
        target=class_id*np.ones((data_item.shape[0],))
        class_id +=1
        labels.append(target)

face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0)

print(face_dataset.shape)
print(face_labels.shape)

while True:
    ret,frame= cap.read()
    if ret==False:
        continue

    faces=face_cascade.detectMultiScale(frame,1.3,5)
    if(len(faces)==0):
        continue

    for face in faces:
        x,y,w,h=face

        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))

        val=knn(face_dataset,face_labels,face_section.flatten())


        pred_name=names[int(val)]
        cv2.putText(frame,(pred_name),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)

    cv2.imshow('face',frame)

    if (cv2.waitKey(1) & 0xFF )== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()