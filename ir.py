import cv2
import sys

# Initialize the Logitech C270 (Try 0 if 1 doesn't work)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

# Create CLAHE object once outside the loop for efficiency
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))

print("Press 'q' to quit. Press 's' to save a screenshot.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # 1. Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Increase Contrast (Normalization)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # 3. CLAHE
    enhanced = clahe.apply(norm)

    # 4. Bilateral Filter (Smooths skin noise, keeps vein edges sharp)
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # 5. Unsharp Mask to sharpen the vein "tubes"
    # Formula: result = original * 1.5 - blurred * 0.5
    gaussian_3 = cv2.GaussianBlur(filtered, (0, 0), 2.0)
    unsharp = cv2.addWeighted(filtered, 1.5, gaussian_3, -0.5, 0)

    # Display the results
    cv2.imshow('Raw IR Feed', gray)
    cv2.imshow('Advanced Vein Enhancement', unsharp)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        sys.exit()
    elif key == ord('s'):
        cv2.imwrite('vein_capture.png', unsharp)
        print("Screenshot saved!")

# Cleanup
cap.release()
cv2.destroyAllWindows()