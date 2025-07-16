import cv2

# Load the pre-trained Haar Cascade for number plate detection
plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect number plates
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in plates:
        # Draw rectangle around plate
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Number Plate", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Optionally, crop and save the number plate
        plate_roi = frame[y:y+h, x:x+w]
        cv2.imshow("Detected Plate", plate_roi)

    # Show full frame
    cv2.imshow("Number Plate Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
