import cv2
import numpy as np

# Initialize the camera (0 for default webcam)
cap = cv2.VideoCapture(0)

# Function to count fingers using convexity defects
def count_fingers(contour, defects):
    count = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            # Get start, end, farthest point, and depth
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            # Calculate the lengths of sides of the triangle formed by the defect
            a = np.linalg.norm(np.array(end) - np.array(start))
            b = np.linalg.norm(np.array(far) - np.array(start))
            c = np.linalg.norm(np.array(end) - np.array(far))

            # Calculate the angle using cosine rule
            angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c + 1e-5)) * 57

            # Count as a finger if the angle is less than 90 and depth is significant
            if angle <= 90 and d > 10000:
                count += 1
    return count

# Function to return a message depending on the number of fingers detected
def get_message(fingers):
    messages = {
        0: "Start",
        1: "Stop",
        2: "Yes",
        3: "No",
        4: "Enter",
        5: "Bye"
    }
    return messages.get(fingers, "")  # Default to empty string if not found

# Main loop to capture frames and process hand gestures
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the image for natural interaction
    roi = frame[100:600, 100:450]  # Define Region of Interest (ROI) for hand detection

    # Draw a rectangle around the ROI
    cv2.rectangle(frame, (100, 100), (450, 450), (0, 255, 0), 2)

    # Convert ROI to HSV for better skin color detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define skin color range and create a mask
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)  # Smooth the mask

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    finger_count = 0
    message = ""

    # Process the largest contour if found
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 3000:  # Ignore small objects
            # Find convex hull and defects
            hull = cv2.convexHull(max_contour, returnPoints=False)
            defects = cv2.convexityDefects(max_contour, hull)

            # Count fingers from convexity defects
            finger_count = count_fingers(max_contour, defects) + 1
            finger_count = min(finger_count, 5)  # Limit to 5 fingers max

            message = get_message(finger_count)  # Get meaning of the gesture

            # Draw the contour on the ROI
            cv2.drawContours(roi, [max_contour], -1, (255, 255, 255), 2)

    # Display finger count and gesture meaning
    cv2.putText(frame, f"Shraddha's Fingers: {finger_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Where gestures meet understanding", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Display usage guide on the screen
    cv2.putText(frame, f"1. Palm fingers closed - [ Stop ]", (300, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"2. Palm fingers closes + thumb open - [ Yes ]", (300, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"3. Star wars Spock's symbol - [ No ]", (300, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"4. Four fingers + open thumb close - [ Enter ]", (300, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"5. Five fingers open - [ Bye ]", (300, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Display interpreted gesture message if available
    if message:
        cv2.putText(frame, f"Sign Meaning : {message}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show the frame in a window
    cv2.imshow("Sign Language Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
