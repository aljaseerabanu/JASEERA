import cv2
import time

# Start video capture from webcam
cap = cv2.VideoCapture(0)
time.sleep(2)  # Allow the camera to warm up

# Read first frame
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    # Compute absolute difference between two frames
    diff = cv2.absdiff(frame1, frame2)

    # Convert to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to create binary image
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image to fill in holes
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours from the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue

        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)

    if motion_detected:
        cv2.putText(frame1, "⚠ Motion Detected!", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame1, "No Motion", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Motion Detection", frame1)

    # Update frames
    frame1 = frame2
    ret, frame2 = cap.read()

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
