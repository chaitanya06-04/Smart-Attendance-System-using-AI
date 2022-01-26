import cv2
import imutils

img=cv2.imread('thanosMarvelMovieimage_28.jpeg')

img1=imutils.resize(img,width=500,height=300)
cv2.imwrite('img2.jpeg',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()