#include <cv.h>
#include <highgui.h>

using namespace cv;

void draw_surf_points(Mat img) {
	Ptr<FeatureDetector> detector = new SurfFeatureDetector(2000);
	vector<KeyPoint> features;

	detector->detect(img, features);

	for (size_t i = 0; i < features.size(); i++) {
		circle(img, features[i].pt, 5, Scalar(0, 0, 0), -1);
	}
}

int main(int argc, char** argv) {
	Mat source_img;
	Mat test_img;
	source_img = imread("images/email1/source.png");
	test_img = imread("images/email1/gmail.png");

	if (!(source_img.data && test_img.data)) {
		printf("No image data\n");
		return -1;
	}

	draw_surf_points(source_img);
	draw_surf_points(test_img);

	namedWindow("w1", CV_WINDOW_AUTOSIZE);
	imshow("w1", source_img);
	namedWindow("w2", CV_WINDOW_AUTOSIZE);
	imshow("w2", test_img);

	waitKey(0);

	return 0;
}

