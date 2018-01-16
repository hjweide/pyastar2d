import numpy as np
import cv2

img = cv2.imread("maze5.png",0)

avging = cv2.blur(img,(10,10))

cv2.imwrite("blurred.png",avging)

cv2.imshow("image", avging)
cv2.waitKey(0)
cv2.destroyAllWindows()

