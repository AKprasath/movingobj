import cv2
import imutils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define variables
first_frame = None
area_threshold = 500  # Adjust this value for sensitivity

while True:
    # Capture frame
    ret, frame = cap.read()

    # Resize frame (optional)
    frame = imutils.resize(frame, width=500)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur for noise reduction
    gaussian_blur = cv2.GaussianBlur(gray, (21, 21), 0)

    # Check for first frame
    if first_frame is None:
        first_frame = gaussian_blur
        continue

    # Calculate absolute difference between frames
    frame_diff = cv2.absdiff(first_frame, gaussian_blur)

    # Threshold for significant changes
    thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)[1]

    # Find contours in thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect motion
    motion_detected = False
    for cnt in contours:
        if cv2.contourArea(cnt) > area_threshold:
            motion_detected = True
            # Optional: Draw bounding box around motion area
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break  # Limit to one detected area (optional)

    # Display text based on motion detection
    text = "Normal" if not motion_detected else "Motion Detected"
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display frame
    cv2.imshow("Motion Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
