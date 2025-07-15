import cv2
import numpy as np

# Step 1: Open the default webcam (0 is usually the built-in camera)
cap = cv2.VideoCapture(0)

# Step 2: Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Step 3: Capture frames in a loop
while True:
    ret, frame = cap.read()  # ret = success flag, frame = captured image
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for yellow color
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a binary mask where yellow colors are white
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image to highlight yellow areas
    yellow_detected = cv2.bitwise_and(frame, frame, mask=mask)

    # Optional: Count yellow pixels to detect presence
    yellow_pixels = cv2.countNonZero(mask)
    if yellow_pixels > 500:  # You can adjust this threshold
        cv2.putText(frame, "Yellow Object Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show original and mask
    cv2.imshow("Webcam Feed", frame)
    cv2.imshow("Yellow Object Detection", yellow_detected)

    # Step 4: Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 5: Release camera and close windows
cap.release()
cv2.destroyAllWindows()
