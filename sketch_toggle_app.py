import cv2
import numpy as np

# Load image
image_path = r"b57d9993-68a3-410a-b13b-a49da527e7c0.png"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("❌ Failed to load image.")

scale_percent = 60
dim = (int(image.shape[1] * scale_percent / 100), int(image.shape[0] * scale_percent / 100))
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_eq = cv2.equalizeHist(gray)

cv2.namedWindow("Sketch Toggle App")
def nothing(x): pass

cv2.createTrackbar("Mode: 0=Gray, 1=Color", "Sketch Toggle App", 0, 1, nothing)
cv2.createTrackbar("Blur", "Sketch Toggle App", 5, 50, nothing)
cv2.createTrackbar("Contrast", "Sketch Toggle App", 230, 300, nothing)

while True:
    mode = cv2.getTrackbarPos("Mode: 0=Gray, 1=Color", "Sketch Toggle App")
    blur_val = cv2.getTrackbarPos("Blur", "Sketch Toggle App")
    contrast_val = cv2.getTrackbarPos("Contrast", "Sketch Toggle App")

    if blur_val < 1: blur_val = 1
    if blur_val % 2 == 0: blur_val += 1

    if mode == 0:
        inverted = 255 - gray_eq
        blurred = cv2.GaussianBlur(inverted, (blur_val, blur_val), 0)
        inverted_blurred = 255 - blurred
        sketch = cv2.divide(gray_eq, inverted_blurred, scale=float(contrast_val))
        cv2.imshow("Sketch Toggle App", sketch)
    else:
        color_smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        block_size = max(3, blur_val | 1)
        edges = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, blockSize=block_size, C=2)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        sketch = cv2.bitwise_and(color_smoothed, edges_colored)
        cv2.imshow("Sketch Toggle App", sketch)

    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite("final_sketch_output.jpg", sketch)
        print("✅ Saved as final_sketch_output.jpg")
        break
    elif key == 27:
        break

cv2.destroyAllWindows()