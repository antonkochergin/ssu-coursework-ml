import cv2

img = cv2.imread("./foots/Test/i.jpg")
img = cv2.imread("./foots/1/IMG_0315.jpg")
img = cv2.resize(img , (img.shape[1] // 4 , img.shape[0]//4))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img , (7,7), 0)
edges = cv2.Canny(img, 100, 125)


# contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# img_contours = img.copy()
# img_filtered = img.copy()
#

contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# cv2.imshow("Original Image", img)
cv2.imshow("Edges (Canny)", edges)
# cv2.imshow("All Contours", img_contours)
# cv2.imshow("Filtered by Area", img_filtered)
cv2.waitKey(0)
