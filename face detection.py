import cv2
import os

# Get the full path to the haarcascade_frontalface_default.xml file
cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

# Open the video capture device (0 corresponds to the default camera)
video = cv2.VideoCapture(0)

# Check if the video capture device is opened successfully
if not video.isOpened():
    print("Error: Could not open video device.")
    exit()

while True:
    # Read a frame from the video capture device
    check, frame = video.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Display the frame with the detected faces
    cv2.imshow("Face Detection", frame)

    # Check for the 'q' key to exit the loop
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture device and close all windows
video.release()
cv2.destroyAllWindows()