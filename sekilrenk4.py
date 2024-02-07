import cv2
import numpy as np

def dynamic_color_threshold(frame, target_color_lower, target_color_upper):
    """Apply dynamic color threshold based on the target color."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, target_color_lower, target_color_upper)
    return mask

def filter_contours(contours, min_area):
    """Filter contours based on minimum area."""
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
    return filtered_contours

def detect_color(hsv_color):
    """Determine color based on HSV values."""
    # Define color ranges in HSV
    blue_range = (np.array([100, 80, 50]), np.array([140, 255, 255]))
    red_range = (np.array([0, 100, 100]), np.array([10, 255, 255]))
    yellow_range = (np.array([20, 100, 100]), np.array([30, 255, 255]))
    green_range = (np.array([40, 100, 100]), np.array([80, 255, 255]))
    orange_range = (np.array([10, 100, 100]), np.array([20, 255, 255]))
    purple_range = (np.array([120, 100, 100]), np.array([160, 255, 255]))
    pink_range = (np.array([160, 100, 100]), np.array([180, 255, 255]))

    if blue_range[0][0] <= hsv_color[0] <= blue_range[1][0] and \
       blue_range[0][1] <= hsv_color[1] <= blue_range[1][1] and \
       blue_range[0][2] <= hsv_color[2] <= blue_range[1][2]:
        return "Blue"
    elif red_range[0][0] <= hsv_color[0] <= red_range[1][0] and \
         red_range[0][1] <= hsv_color[1] <= red_range[1][1] and \
         red_range[0][2] <= hsv_color[2] <= red_range[1][2]:
        return "Red"
    elif yellow_range[0][0] <= hsv_color[0] <= yellow_range[1][0] and \
         yellow_range[0][1] <= hsv_color[1] <= yellow_range[1][1] and \
         yellow_range[0][2] <= hsv_color[2] <= yellow_range[1][2]:
        return "Yellow"
    elif green_range[0][0] <= hsv_color[0] <= green_range[1][0] and \
         green_range[0][1] <= hsv_color[1] <= green_range[1][1] and \
         green_range[0][2] <= hsv_color[2] <= green_range[1][2]:
        return "Green"
    elif orange_range[0][0] <= hsv_color[0] <= orange_range[1][0] and \
         orange_range[0][1] <= hsv_color[1] <= orange_range[1][1] and \
         orange_range[0][2] <= hsv_color[2] <= orange_range[1][2]:
        return "Orange"
    elif purple_range[0][0] <= hsv_color[0] <= purple_range[1][0] and \
         purple_range[0][1] <= hsv_color[1] <= purple_range[1][1] and \
         purple_range[0][2] <= hsv_color[2] <= purple_range[1][2]:
        return "Purple"
    elif pink_range[0][0] <= hsv_color[0] <= pink_range[1][0] and \
         pink_range[0][1] <= hsv_color[1] <= pink_range[1][1] and \
         pink_range[0][2] <= hsv_color[2] <= pink_range[1][2]:
        return "Pink"
    else:
        return "Undefined"

def detect_shape_and_color(contour, hsv_frame):
    """Determine the shape and color of the given contour."""
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    sides = len(approx)

    color = detect_color(hsv_frame[int(contour.mean(axis=0)[:, 1]), int(contour.mean(axis=0)[:, 0])])

    if sides == 3:
        return "Triangle", color
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h

        if 0.8 <= aspect_ratio <= 1.2:
            return "Square", color
        else:
            return "Rectangle", color
    elif sides == 5:
        return "Pentagon", color
    elif sides == 6:
        return "Hexagon", color
    else:
        (x, y), radius = cv2.minEnclosingCircle(approx)
        circularity = cv2.contourArea(approx) / (np.pi * radius ** 2)

        return "Circle", color if circularity >= 0.6 else "Undefined"

def process_frame(frame):
    """Process each frame to detect shapes and colors."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define target color ranges in HSV for different colors
    blue_lower = np.array([100, 80, 50])
    blue_upper = np.array([140, 255, 255])

    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    green_lower = np.array([40, 100, 100])
    green_upper = np.array([80, 255, 255])

    orange_lower = np.array([10, 100, 100])
    orange_upper = np.array([20, 255, 255])

    purple_lower = np.array([120, 100, 100])
    purple_upper = np.array([160, 255, 255])

    pink_lower = np.array([160, 100, 100])
    pink_upper = np.array([180, 255, 255])

    # Apply color thresholding for each color
    blue_mask = dynamic_color_threshold(frame, blue_lower, blue_upper)
    red_mask = dynamic_color_threshold(frame, red_lower, red_upper)
    yellow_mask = dynamic_color_threshold(frame, yellow_lower, yellow_upper)
    green_mask = dynamic_color_threshold(frame, green_lower, green_upper)
    orange_mask = dynamic_color_threshold(frame, orange_lower, orange_upper)
    purple_mask = dynamic_color_threshold(frame, purple_lower, purple_upper)
    pink_mask = dynamic_color_threshold(frame, pink_lower, pink_upper)

    # Combine masks for different colors
    mask = cv2.bitwise_or(cv2.bitwise_or(blue_mask, red_mask, yellow_mask),
                          cv2.bitwise_or(green_mask, orange_mask, purple_mask, pink_mask))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = filter_contours(contours, min_contour_area)

    for contour in filtered_contours:
        shape, color = detect_shape_and_color(contour, hsv_frame)

        if color != "Undefined":  # If color is defined and not "Undefined," only then draw
            x, y, w, h = cv2.boundingRect(contour)

            if shape != "Circle":
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if shape == "Circle":
                cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), int((w + h) / 4), (255, 0, 0), 2)

            cv2.putText(frame, f"{color} {shape}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

# Open the camera connection
cap = cv2.VideoCapture(0)

# Set frame dimensions (1920x1080)
cap.set(3, 1920)
cap.set(4, 1080)

# Set minimum contour area
min_contour_area = 50 # If you make this  too small, it will pick up a lot of noise as 'blobs' instead of actual objects to 50 pixels squared

while True:
    ret, frame = cap.read()

    processed_frame = process_frame(frame)
    cv2.imshow("Object Detection", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
