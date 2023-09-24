# FaceFolio: Smart Attendance  (optimized using Threads)
Attendance marking in a classroom can be a tedious and time-consuming task, especially when dealing with many students. Marking attendance by following  traditional methods has been known to lack reliability
The entire procedure is optimized using threads so that the old frames can be processed in parallel and the new frames can be continuously captured. The proposed system utilizes facial recognition technology to automate the process of attendance marking, resulting in a reduction of workload for educators compared to traditional methods. Implementing a software-based attendance system for any organization lowers the hardware cost, which is a great accomplishment and greatly benefits society. This is more affordable and practical than any other biometric system. The entire system is automated, and hence no additional staff is required to monitor the attendance procedure. The implemented system not only lowers costs but also increases operational efficiency. Since time plays an important factor in any organization, the project aims at preventing time loss and providing it for other productive purposes. Manual attendance systems can be prone to human errors, as in the case of institutes, there are many cases of pseudo-attendance commonly called ‘proxy.’ A refined version of the project will deliver satisfactory accuracy with minimal errors. The traditional methods 
of attendance marking are time-consuming and need additional staff. Moreover, the data can easily be manipulated.

## Software Requirements
● The project will use the Python programming language.
● Advanced Python Classifiers like OpenCV will be used to detect the face of the student in front of the camera.
● The software must be capable of marking the attendance of the student based on the detected face.
● The software must feature a Graphical User Interface (GUI) to display the attendance being recorded.
● The GUI should display attendance in real-time as the software detects the student's face.
● The attendance will be stored in an Excel sheet for further analysis and record keeping.
● The software should be able to detect multiple faces simultaneously in a crowded environment.
● The software should be able to operate in different lighting conditions and camera angles.
● The accuracy and reliability of the software must be tested and validated.
● The software should be user-friendly and easy to use for teachers and administrative staff

## System Design
This project makes use of the built-in Python library "face_recognition" whose foundation is a well-known deep learning model. In this work, we build a facial recognition system to record and store employee attendance. The intention is to use the OpenCV module to capture video frames and then encode them using face_recognition module so that they match the registered faces. The matching criteria use similarity scores and distances in order to predict the person. If an employee is identified, their registration number is displayed; otherwise, it says "Unknown" if they are not.  The entire procedure is optimised using threads so that the old frames can be processed in parallel and the new frames can be continuously captured. The time stamp and the marked attendance are automatically saved in a csv file. The work is capable of using GPU power to speed up the process. Duplicate entries can be handled automatically. The current work can identify up to three faces at once, but if more computing power is available, it can identify more faces. The project has a user -friendly interface for its easier use. 

## Results

![image](https://github.com/pulak2002/Automatic-Attendance-Marking-Using-Facial-Recognition-Technique/assets/110912267/9c731a15-dab8-4cfc-a67d-ff5bdfb65733)

![image](https://github.com/pulak2002/Automatic-Attendance-Marking-Using-Facial-Recognition-Technique/assets/110912267/eff95e6f-b800-407b-838b-9292a5923334)

![image](https://github.com/pulak2002/Automatic-Attendance-Marking-Using-Facial-Recognition-Technique/assets/110912267/8252502f-35b9-4efb-a413-3a170279a107)

![image](https://github.com/pulak2002/Automatic-Attendance-Marking-Using-Facial-Recognition-Technique/assets/110912267/2ebb3b94-6256-4e69-9de7-c8168e570575)

![image](https://github.com/pulak2002/Automatic-Attendance-Marking-Using-Facial-Recognition-Technique/assets/110912267/3f81ac52-81db-4c7d-bac7-ec376a092560)

![image](https://github.com/pulak2002/Automatic-Attendance-Marking-Using-Facial-Recognition-Technique/assets/110912267/87195ee2-981c-4c90-9506-791eba550b34)

