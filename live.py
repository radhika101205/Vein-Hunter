import cv2
import numpy as np
from skimage.filters import frangi

def live_vein_master_pipeline():
    video_url = "http://admin:admin@172.31.91.13:8081" 
    
    print(f"Bypassing Windows... connecting to {video_url}")
    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        print("Error: Could not connect.")
        return

    cv2.namedWindow("Live Vein Tuner")
    def nothing(x): pass
    cv2.createTrackbar("Threshold", "Live Vein Tuner", 15, 100, nothing)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    history_length = 5
    frame_history = []

    print("Starting Ultimate Vein Scanner...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        box_size = 350
        x = int((width - box_size) / 2)
        y = int((height - box_size) / 2)

        cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), (255, 255, 255), 2)
        roi = frame[y:y+box_size, x:x+box_size]

        # --- 1. CHANNEL EXTRACTION & AVERAGING ---
        b, green_channel, r = cv2.split(roi)
        frame_history.append(green_channel)
        if len(frame_history) > history_length:
            frame_history.pop(0)
        avg_signal = np.mean(frame_history, axis=0).astype(np.uint8)

        # --- 2. THE "DULL RAZOR" (Bring back the hair remover!) ---
        # We must mathematically shave the hair before enhancing contrast
        hair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hairless = cv2.morphologyEx(avg_signal, cv2.MORPH_CLOSE, hair_kernel)

        # --- 3. CONTRAST BOOST ---
        norm = cv2.normalize(hairless, None, 0, 255, cv2.NORM_MINMAX)
        enhanced = clahe.apply(norm)

        # --- 4. HUMAN VISUALIZATION FEED (Sharp & Edgy) ---
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        gaussian_3 = cv2.GaussianBlur(filtered, (0, 0), 2.0)
        unsharp = cv2.addWeighted(filtered, 1.5, gaussian_3, -0.5, 0)
        
        # Show the user the cool sharp image
        cv2.imshow("Your Preprocessing Pipeline", unsharp)

        # --- 5. THE MACHINE MATH FEED (Soft & Broad) ---
        # Frangi hates high-frequency noise. We melt the unsharp mask into broad gradients.
        frangi_input = cv2.GaussianBlur(unsharp, (15, 15), 0)

        # --- 6. FRANGI (Upgraded Scale Space) ---
        # CRITICAL FIX: Increased sigmas (8 to 24) to look for thick veins, completely ignoring thin wrinkles.
        vein_probabilities = frangi(frangi_input, sigmas=np.arange(8, 24, 4), beta=0.5, gamma=15, black_ridges=True)
        vein_prob_normalized = cv2.normalize(vein_probabilities, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # --- 7. TUNING & FILTERING ---
        current_threshold = cv2.getTrackbarPos("Threshold", "Live Vein Tuner")
        _, binary_veins = cv2.threshold(vein_prob_normalized, current_threshold, 255, cv2.THRESH_BINARY)

        # Contour filtering (Kill any remaining wrinkle speckles under 80 pixels)
        contours, _ = cv2.findContours(binary_veins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(binary_veins)
        for contour in contours:
            if cv2.contourArea(contour) > 80: 
                cv2.drawContours(clean_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Bridging
        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        final_veins = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, bridge_kernel)

        # --- 8. OVERLAY ---
        roi[final_veins == 255] = [0, 255, 0]
        frame[y:y+box_size, x:x+box_size] = roi

        cv2.imshow("Live Vein Tuner", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('vein_capture.png', frame)
            print("Screenshot saved to folder!")

    cap.release()
    cv2.destroyAllWindows()

live_vein_master_pipeline()