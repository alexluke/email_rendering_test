#include <cv.h>
#include <highgui.h>

using namespace cv;

Rect getBorder(Mat img) {
	Mat grayImg, tmpImg;
	cvtColor(img, grayImg, CV_BGR2GRAY);
	threshold(grayImg, tmpImg, 150, 255, THRESH_BINARY);

	// FIXME: This is super inefficient
	int topRow = 0;
	int bottomRow = tmpImg.rows - 1;
	int leftCol = 0;
	int rightCol = tmpImg.cols - 1;
	// Skip edge pixels because the image might have a one pixel border
	for (int row = 1; row < tmpImg.rows - 1; row++) {
		unsigned char* p = tmpImg.ptr<unsigned char>(row);
		bool nonWhite = false;
		// Skip the edge pixels
		for (int col = 1; col < tmpImg.cols - 1; col++) {
			p++;
			if (*p == 0) {
				nonWhite = true;
			}
		}
		if (nonWhite) {
			topRow = row - 1;
			break;
		}
	}

	// Skip edge pixels because the image might have a one pixel border
	for (int row = tmpImg.rows - 2; row > 1; row--) {
		unsigned char* p = tmpImg.ptr<unsigned char>(row);
		bool nonWhite = false;
		// Skip the edge pixels
		for (int col = 1; col < tmpImg.cols - 1; col++) {
			p++;
			if (*p == 0) {
				nonWhite = true;
			}
		}
		if (nonWhite) {
			bottomRow = row + 1;
			break;
		}
	}

	// Skip edge pixels because the image might have a one pixel border
	for (int col = 1; col < tmpImg.cols - 1; col++) {
		unsigned char* p = tmpImg.ptr<unsigned char>(1) + col;
		bool nonWhite = false;
		// Skip the edge pixels
		for (int row = 1; row < tmpImg.rows - 1; row++) {
			if (*p == 0) {
				nonWhite = true;
			}
			p += tmpImg.cols;
		}
		if (nonWhite) {
			leftCol = col - 1;
			break;
		}
	}

	// Skip edge pixels because the image might have a one pixel border
	for (int col = tmpImg.cols - 2; col > 1; col--) {
		unsigned char* p = tmpImg.ptr<unsigned char>(1) + col;
		bool nonWhite = false;
		// Skip the edge pixels
		for (int row = 1; row < tmpImg.rows - 1; row++) {
			if (*p == 0) {
				nonWhite = true;
			}
			p += tmpImg.cols;
		}
		if (nonWhite) {
			rightCol = col + 1;
			break;
		}
	}

	return Rect(leftCol, topRow, rightCol - leftCol, bottomRow - topRow);
}

int main(int argc, char** argv) {
	Mat srcImg, dstImg;
	srcImg = imread("images/email1/source.png");
	dstImg = imread("images/email1/gmail.png");

	if (!(srcImg.data && dstImg.data)) {
		printf("No image data\n");
		return -1;
	}

	Rect rect = getBorder(srcImg);

	srcImg = srcImg(rect);

	SurfFeatureDetector detector(2000);
	vector<KeyPoint> srcFeatures, dstFeatures;

	detector.detect(srcImg, srcFeatures);
	detector.detect(dstImg, dstFeatures);

	SurfDescriptorExtractor extractor;
	Mat srcDescriptors, dstDescriptors;
	extractor.compute(srcImg, srcFeatures, srcDescriptors);
	extractor.compute(dstImg, dstFeatures, dstDescriptors);

	BruteForceMatcher< L2<float> > matcher;
	vector<DMatch> matches;
	matcher.match(srcDescriptors, dstDescriptors, matches);

	vector<int> pairOfSrcKP(matches.size()), pairOfDstKP(matches.size());
	for (size_t i = 0; i < matches.size(); i++) {
		pairOfSrcKP[i] = matches[i].queryIdx;
		pairOfDstKP[i] = matches[i].trainIdx;
	}

	vector<Point2f> srcPoints, dstPoints;
	KeyPoint::convert(srcFeatures, srcPoints, pairOfSrcKP);
	KeyPoint::convert(dstFeatures, dstPoints, pairOfDstKP);

	Mat src2DFeatures, dst2DFeatures;
	Mat(srcPoints).copyTo(src2DFeatures);
	Mat(dstPoints).copyTo(dst2DFeatures);

	vector<uchar> outlierMask;
	Mat H;
	H = findHomography(src2DFeatures, dst2DFeatures, outlierMask, RANSAC, 3);

	Mat matchImg;
	drawMatches(srcImg, srcFeatures, dstImg, dstFeatures, matches, matchImg, Scalar::all(-1), Scalar::all(-1),
		reinterpret_cast<const vector<char>&>(outlierMask));

	Mat alignedDstImg;
	warpPerspective(dstImg, alignedDstImg, H.inv(), srcImg.size(), INTER_LINEAR, BORDER_CONSTANT);

	vector<Point2f> objCorners(4);
	objCorners[0] = cvPoint(0, 0);
	objCorners[1] = cvPoint(srcImg.cols, 0);
	objCorners[2] = cvPoint(srcImg.cols, srcImg.rows);
	objCorners[3] = cvPoint(0, srcImg.rows);

	vector<Point2f> sceneCorners(4);
	perspectiveTransform(objCorners, sceneCorners, H);

	line(dstImg, sceneCorners[0], sceneCorners[1], Scalar(0, 255, 0), 4);
	line(dstImg, sceneCorners[1], sceneCorners[2], Scalar(0, 255, 0), 4);
	line(dstImg, sceneCorners[2], sceneCorners[3], Scalar(0, 255, 0), 4);
	line(dstImg, sceneCorners[3], sceneCorners[0], Scalar(0, 255, 0), 4);

	Mat differenceImg;
	absdiff(srcImg, alignedDstImg, differenceImg);

	double n;
	n = norm(srcImg, alignedDstImg);
	printf("%f\n", n);

	imshow("Matches: Src image (left) to dst (right)", matchImg);
	imshow("Original", srcImg);
	imshow("Matched", dstImg);
	imshow("Aligned", alignedDstImg);
	imshow("Difference", differenceImg);

	waitKey(0);

	return 0;
}

