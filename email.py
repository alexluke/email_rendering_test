import cv

def draw_surf_points(img):
	keypoints, descriptors = cv.ExtractSURF(img, None, cv.CreateMemStorage(), (0, 2000, 3, 4))

	print len(keypoints)
	for (x, y), laplacian, size, d, hessian in keypoints:
		x = int(x)
		y = int(y)
		cv.Circle(img, (x, y), 5, cv.CV_RGB(0, 0, 0), cv.CV_FILLED)

source_img = cv.LoadImageM('images/email1/source.png', cv.CV_LOAD_IMAGE_GRAYSCALE)
test_img = cv.LoadImageM('images/email1/gmail.png', cv.CV_LOAD_IMAGE_GRAYSCALE)

draw_surf_points(source_img)
draw_surf_points(test_img)


cv.NamedWindow("w1")
cv.ShowImage("w1", source_img)
cv.NamedWindow("w2")
cv.ShowImage("w2", test_img)

cv.WaitKey()
