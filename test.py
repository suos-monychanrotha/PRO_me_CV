import cv2

# Force DirectShow backend on Windows if needed, otherwise just use cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera device.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab a frame.")
        break
        
    cv2.imshow('Camera Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
