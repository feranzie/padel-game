import cv2

# List to store clicked points
clicked_points = []
nos=1045
# Mouse callback function to capture points
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked point: ({x}, {y})")
        clicked_points.append((x, y))  # Store the clicked point
        # Draw a circle at the clicked point
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        #Annotate the coordinates on the image
        if 142 <= y < 213:
            x_meter = round((x - 858) / 90.6, 2)
            y_meter = round((y - 144) * 3 / 69, 2)
        elif 213 <= y < 498:
            x_meter = round((x - 622) / 135.5, 2)
            y_meter = round((y - 213) * 7 / 285 + 3, 2)
        elif 498 <= y < 842:
            x_meter = round((x - 391) / 181.8, 2)
            y_meter = round((y - 494) * 3.5 / 761 + 10, 2)
        elif 842 <= y < 1259:
            x_meter = round((x - 161) / 225.3, 2)
            y_meter = round((y - 827) * 3.5 / 432 + 13.5, 2)
        elif 1259 <= y < 1435:
            x_meter = round((x - 72) / 240.9, 2)
            y_meter = round((y - 1238) * 3 / 394 + 17, 2)

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

