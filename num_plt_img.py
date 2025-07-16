import cv2

# Load image from local file
image = cv2.imread("num_plate.jpeg")  # Replace with your image file

# Load Haar Cascade for number plate
plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect number plates in the image
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Loop through all detected plates
for (x, y, w, h) in plates:
    # Draw rectangle around the plate
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "Number Plate", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Crop and save the number plate
    plate = image[y:y+h, x:x+w]
    cv2.imwrite("detected_plate.jpg", plate)
    cv2.imshow("Detected Plate", plate)

# Show the final image
cv2.imshow("Number Plate Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
