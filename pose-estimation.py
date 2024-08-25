import cv2


print("hello")
# Load an image
img = cv2.imread('erg.jpg')

# Display the image
cv2.imshow('Test Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
