import cv2

# List to store clicked points
clicked_points = []
nos=1045
pixel_to_meter=1/135
# Mouse callback function to capture points
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked point: ({x}, {y})")
        clicked_points.append((x, y))  # Store the clicked point
        # Draw a circle at the clicked point
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        #Annotate the coordinates on the image
        if 142 <= y < 213:
            x_meter = x * pixel_to_meter*1.59
            y_meter = y * pixel_to_meter* 5.87
        elif 213 <= y < 498:
            x_meter = x * pixel_to_meter*1.195
            y_meter = x * pixel_to_meter*3.32
        elif 498 <= y < 842:
            x_meter = x * pixel_to_meter*0.85
            y_meter = y * pixel_to_meter*1.37
        elif 842 <= y < 1259:
            x_meter = x * pixel_to_meter*0.66
            y_meter = y * pixel_to_meter* 1.13
        elif 1259 <= y < 1435:
            x_meter = x * pixel_to_meter*0.59
            y_meter = y * pixel_to_meter*1.13
        cv2.putText(frame, f'({x_meter }, {y_meter})', (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
       
        cv2.imshow('Image', frame)

# Load an image or video frame
frame = cv2.imread(f'output_frame{nos}.jpg')  # Replace with your image path
# Create a window with resizable option
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', frame.shape[1], frame.shape[0])  # Set the window size to match the image size

# Set the mouse callback function
cv2.setMouseCallback('Image', click_event)

# Display the image
cv2.imshow('Image', frame)

# Wait until the user presses a key or closes the window
cv2.waitKey(0)

# Save the new image with annotations
cv2.imwrite(f'vframev{nos}.png', frame)  # Save with your desired file name

cv2.destroyAllWindows()

