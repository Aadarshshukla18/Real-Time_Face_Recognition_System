🧠 Real-Time Face Detection System using OpenCV

This project demonstrates a real-time face detection system built using Python and OpenCV. It captures live video from your webcam, processes each frame, and detects human faces using the Haar Cascade Classifier. Detected faces are highlighted with green bounding boxes, offering a simple yet powerful introduction to computer vision.

🚀 Features

✔ Real-time webcam video capture
✔ Face detection using Haar Cascade
✔ Live bounding box visualization
✔ Lightweight & easy to run
✔ Perfect for beginners in AI/ML and OpenCV

🛠️ Technologies Used

Python
OpenCV (cv2)
Haar Cascade Classifier

📸 How It Works (Explanation)

Load the Haar Cascade model for face detection
Open the webcam using cv2.VideoCapture(0)
Convert each video frame to grayscale
Detect faces with detectMultiScale()
Draw bounding boxes around detected faces
Display the live output window
Press ‘a’ to exit

🧾 Code Snippet
face_cap = cv2.CascadeClassifier("C:/Users/adish/anaconda3/anaconda/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(col, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("video_live", video_data)

    if cv2.waitKey(10) == ord("a"):
        break

video_cap.release()
cv2.destroyAllWindows()

▶️ How to Run

Install dependencies
pip install opencv-python
Run the script
python face_detection.py
Allow webcam access
Press 'a' to close the window

📌 Applications

Security & surveillance
Attendance systems
Smart door locks
Human-computer interaction
Beginner computer vision projects

📄 License

This project is open-source and free to use for learning and development.
