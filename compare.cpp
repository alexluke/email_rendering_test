#include <cv.h>
#include <highgui.h>

using namespace cv;

int main(int argc, char** argv) {
	Mat source_img;
	Mat test_img;
	source_img = imread("images/email1/source.png", CV_LOAD_IMAGE_GRAYSCALE);
	test_img = imread("images/email1/gmail.png", CV_LOAD_IMAGE_GRAYSCALE);

	if (!(source_img.data && test_img.data)) {
		printf("No image data\n");
		return -1;
	}

	SurfFeatureDetector detector(400);
	vector<KeyPoint> keypoints1, keypoints2;

	detector.detect(source_img, keypoints1);
	detector.detect(test_img, keypoints2);

	Mat img_keypoints1, img_keypoints2;

	drawKeypoints(source_img, keypoints1, img_keypoints1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(test_img, keypoints2, img_keypoints2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	imshow("Keypoints 1", img_keypoints1);
	imshow("Keypoints 2", img_keypoints2);

	waitKey(0);

	return 0;
}

