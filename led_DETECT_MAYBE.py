import cv2
import numpy as np

def track_colors(img, colors):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert image from BGR to HSV color space
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    contours = []
    for color in colors:
        lower, upper = color["lower"], color["upper"]
        mask_color = cv2.inRange(hsv, lower, upper)
        mask_color = cv2.bitwise_and(mask_color, mask_color, mask=cv2.inRange(hsv, (0, 30, 0), (180, 255, 255))) # Filter out low value and saturation regions
        contours_color, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours += contours_color
        mask += mask_color
    return mask, contours

colors = [
  
    {"name": "red_ight", "lower": (153, 186, 201), "upper": (255, 255, 255)}, # Adjust yellow color range to filter out low value and saturation regions
]

cap = cv2.VideoCapture(1)

while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()
    
    # Apply color tracking and get the mask and contours
    mask, contours = track_colors(frame, colors)

    # Iterate over the contours and draw bounding boxes around them on the original frame
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        
        color_name = None
        for color in colors:
                lower, upper = color["lower"], color["upper"]
                mask_color = cv2.inRange(frame[y:y+h, x:x+w], lower, upper)
                if np.count_nonzero(mask_color) > 0.3 * mask_color.size:
                    color_name = color["name"]
                    break
        if color_name is not None:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, color_name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Color tracking", frame)

    # Break the loop when the "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
