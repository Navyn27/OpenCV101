import pytesseract
from pytesseract import Output
import PIL.Image
import cv2
import numpy as np

# Page segmentation modes:
#   0    Orientation and script detection (OSD) only.
#   1    Automatic page segmentation with OSD.
#   2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
#   3    Fully automatic page segmentation, but no OSD. (Default)
#   4    Assume a single column of text of variable sizes.
#   5    Assume a single uniform block of vertically aligned text.
#   6    Assume a single uniform block of text.
#   7    Treat the image as a single text line.
#   8    Treat the image as a single word.
#   9    Treat the image as a single word in a circle.
#  10    Treat the image as a single character.
#  11    Sparse text. Find as much text as possible in no particular order.
#  12    Sparse text with OSD.
#  13    Raw line. Treat the image as a single text line,
#        bypassing hacks that are Tesseract-specific.

# OCR Engine modes:
#   0    Legacy engine only.
#   1    Neural nets LSTM engine only.
#   2    Legacy + LSTM engines.
#   3    Default, based on what is available.


myconfig = r"--psm 3 --oem 3"

# text = pytesseract.image_to_string(PIL.Image.open("signs.jpeg"), config=myconfig)
# print(text)
# Read the image in grayscale
image = cv2.imread('meter-img.jpeg', 0)

# Apply Gaussian blur to the image to reduce noise and make edges smoother
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# img = cv2.Canny(image, 50, 60)
blurred = cv2.GaussianBlur(image, (3, 5), 0)

# Crop the image using the specified ROI
# roi_coordinates = [50:200, 100:300]

# Use morphological operations to close gaps and join edges
kernel = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

# Optionally, you can apply additional smoothing with Gaussian blur
smoothed = cv2.GaussianBlur(closed, (5, 5), 0)
cropped_image = smoothed[0:500, 320:1100]
height, width = cropped_image.shape[:2]

# ---------------
# To put a box around each character

boxes = pytesseract.image_to_boxes(cropped_image, config=myconfig)
print(boxes)

for box in boxes.splitlines():
    box = box.split(" ")
    cropped_image = cv2.rectangle(cropped_image, (int(box[1]), height - int(box[2])), (int(box[3]), height - int(box[4])), [0,255,0])

cv2.imshow("img", cropped_image);
cv2.waitKey(0);


# ----------------
#To put boxes around a word

# data = pytesseract.image_to_data(img, config=myconfig, output_type=Output.DICT)

# print(data.keys())
# print(data['text'])

# amount_boxes = len(data['text'])
# for i in range(amount_boxes):
#     if float(data['conf'][i]) > 25: 
#         (x, y, width, height) = (data['left'][i],data['top'][i],data['width'][i],data['height'][i])
#         img = cv2.rectangle(img, (x,y), (x+width,y+height), (0,255,0))
#         img = cv2.putText(img, data['text'][i], (x, y+height+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2,cv2.LINE_AA)
# cv2.imshow("img", img);
# cv2.waitKey(0);
