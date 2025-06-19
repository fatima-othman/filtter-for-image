import cv2
import numpy as np

def edge_detection(frame):
    edges = cv2.Canny(frame, 100, 200)
    return edges

def grayscale_quantization(frame, levels):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    quantized = np.round(gray * levels / 255) * (255 / levels)
    quantized = quantized.astype(np.uint8)
    return quantized

def contrast_enhancement(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    return enhanced

def soft_appearance(frame):
    blurred = cv2.GaussianBlur(frame, (11, 11), 5)
    return blurred

def cartoon_filter(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    gray_blur = cv2.medianBlur(gray, 5)
    
   
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, blockSize=9, C=2)
   
    smoothed = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
    

    def quantize_colors(image, k=8):
        Z = image.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        return res.reshape(image.shape)
    
    quantized = quantize_colors(smoothed, k=8)
    
    cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)
    
    return cartoon

def negative_filter(frame):
    negative = 255 - frame
    return negative

def sepia_filter(frame):
    frame = frame.astype(np.float32)
    sepia_matrix = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia = cv2.transform(frame, sepia_matrix)
    sepia = np.clip(sepia, 0, 255)
    return sepia.astype(np.uint8)

def face_smoothing_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Cannot load Haar cascade file!")
        return frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    smoothed_frame = frame.copy()
    for (x, y, w, h) in faces:
        face_region = smoothed_frame[y:y+h, x:x+w]
        smoothed_face = cv2.bilateralFilter(face_region, d=9, sigmaColor=75, sigmaSpace=75)
        smoothed_frame[y:y+h, x:x+w] = smoothed_face
    return smoothed_frame


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera!")
    exit()

mode = 0 
quantization_levels = 4

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame!")
        break

    if mode == 1:
        processed_frame = edge_detection(frame)
    elif mode == 2:
        processed_frame = grayscale_quantization(frame, quantization_levels)
    elif mode == 3:
        processed_frame = contrast_enhancement(frame)
    elif mode == 4:
        processed_frame = soft_appearance(frame)
    elif mode == 5:
        processed_frame = cartoon_filter(frame)
    elif mode == 6:
        processed_frame = negative_filter(frame)
    elif mode == 7:
        processed_frame = sepia_filter(frame)
    elif mode == 8:
        processed_frame = face_smoothing_filter(frame)
    else:
        processed_frame = frame 

    
    cv2.imshow('Video Processing', processed_frame)

   
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        mode = 1
        print("Mode: Edge Detection")
    elif key == ord('2'):
        mode = 2
      
        try:
            levels_input = input("Enter the number of grayscale levels (e.g., 4, 8, 16): ")
            quantization_levels = int(levels_input)
            if quantization_levels <= 0:
                print("Please enter a positive number. Using default value (4).")
                quantization_levels = 4
            print(f"Mode: Grayscale Quantization with {quantization_levels} levels")
        except ValueError:
            print("Invalid input! Using default value (4).")
            quantization_levels = 4
    elif key == ord('3'):
        mode = 3
        print("Mode: Contrast Enhancement")
    elif key == ord('4'):
        mode = 4
        print("Mode: Soft Appearance")
    elif key == ord('5'):
        mode = 5
        print("Mode: Cartoon Filter")
    elif key == ord('6'):
        mode = 6
        print("Mode: Negative Filter")
    elif key == ord('7'):
        mode = 7
        print("Mode: Sepia Filter")
    elif key == ord('8'):
        mode = 8
        print("Mode: Face Smoothing Filter")
    elif key == ord('q'):
        print("Closing camera and windows...")
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(500)
print("Camera and windows closed successfully!")