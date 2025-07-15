import cv2

# Step 1: Open the default webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Step 2: Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Step 3: Check if webcam is opened successfully
if not cap.isOpened():
    print("❌ Error: Could not access the webcam.")
    exit()

# Step 4: Read frames in a loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame from webcam.")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with rectangles
    cv2.imshow("Webcam Face Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 5: Release resources
cap.release()
cv2.destroyAllWindows()
