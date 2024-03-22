import cv2

# Load images
image1 = cv2.imread('images/lm.png')
image2 = cv2.imread('images/l.png')
image3 = cv2.imread('images/m.png')
image4 = cv2.imread('images/r.png')
image5 = cv2.imread('images/rm.png')

# Create a stitcher object
stitcher = cv2.Stitcher_create()
status, result = stitcher.stitch((image1, image2, image3, image4, image5))

if status == cv2.Stitcher_OK:
    # Crop the top and bottom
    cropped_image = result[10:-45, :]

    # Display cropped image
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save cropped image
    cv2.imwrite('stitched_image_cropped.png', cropped_image)
else:
    print("Stitching failed")
