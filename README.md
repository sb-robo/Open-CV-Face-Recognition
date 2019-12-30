# Open-CV-Face-Recognition
Real-time face recognition project with Open-CV and Python

### Hi there!
- This is a real-ltime face recognition system which can detect Face and also it can recpgnise Face. 
(Recognise only when the sample data of that face has already in <b>Data</b> folder)
- There are two Python file.

#### 1. facedata_collect.py
This python file collects data and stores it in <b>data</b> folder.
Capture atleast 20 sample data for better performance.
- <b>haarcascade_frontalface_alt.xml</b> used for face detection.
#### 2. face_detect.py
This pyhton file detect and recognise face. 
- I have used KNN(K-Nearest Neighbors) Algorithm as Recognizer.
- - I have used <b>Euclidean Distance</b> for distance metric of continuous variables.
- <b>haarcascade_frontalface_alt.xml</b> used for face detection.

#### Process of Face_Recognition System:
- Phase 1: Data gathering
- - Gather face data(images) of the person to be identified.
- - Stores the data into dataset(data folder).
- Phase 2: Train the recognizer
- - Feed the data and id of each face to the recognizer so it can learn.
- Phase 3: Recognition
- - Recognize faces.

### Prerequisites
- Python 3.6 or higher version must be installed in the System where you are running.
- OpenCV must be installed.
- Need atleast one configured camera.

#### Useful Links:
- KNN Algorithm: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm#Parameter_selection
- OpenCV: https://opencv.org/
- haarcascade_classifier: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
