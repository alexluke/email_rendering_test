#include <cv.h>
#include <highgui.h>

using namespace cv;

void printr(Rect rect) {
	printf("Rect at (%d %d). %dx%d\n", rect.x, rect.y, rect.width, rect.height);
}

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
	Mat srcImg, dstImg, originalSrcImg;
	originalSrcImg = imread("images/email1/source.png");
	dstImg = imread("images/email1/gmail.png");

	if (!(originalSrcImg.data && dstImg.data)) {
		printf("No image data\n");
		return -1;
	}

	Rect srcRect = getBorder(originalSrcImg);

	srcImg = originalSrcImg(srcRect);

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
	objCorners[0] = Point(0, 0);
	objCorners[1] = Point(srcImg.cols, 0);
	objCorners[2] = Point(srcImg.cols, srcImg.rows);
	objCorners[3] = Point(0, srcImg.rows);

	vector<Point2f> sceneCorners(4);
	perspectiveTransform(objCorners, sceneCorners, H);

	int top, left, bottom, right;
	top = min(sceneCorners[0].y, sceneCorners[1].y);
	bottom = max(sceneCorners[2].y, sceneCorners[3].y);
	left = min(sceneCorners[0].x, sceneCorners[3].x);
	right = max(sceneCorners[1].x, sceneCorners[2].x);
	Rect dstRect = Rect(left, top, right - left, bottom - top);

	printr(dstRect);
	//dstRect = dstRect + Size(10, 10) - Point(5, 5);
	printr(dstRect);

	printr(srcRect);
	Point p = dstRect.tl() - Point(sceneCorners[0].x, sceneCorners[0].y);
	printf("(%d, %d)\n", p.x, p.y);
	srcRect.x += p.x;
	srcRect.y += p.y;
	srcRect.width = dstRect.width;
	srcRect.height = dstRect.height;
	printr(srcRect);

	srcImg = originalSrcImg(srcRect);

	Mat croppedDstImg;
	dstImg.copyTo(croppedDstImg);
	croppedDstImg = croppedDstImg(dstRect);

	//printr(srcRect);

	rectangle(dstImg, dstRect, Scalar(255, 0, 0), 2);

	line(dstImg, sceneCorners[0], sceneCorners[1], Scalar(0, 255, 0), 2);
	line(dstImg, sceneCorners[1], sceneCorners[2], Scalar(0, 255, 0), 2);
	line(dstImg, sceneCorners[2], sceneCorners[3], Scalar(0, 255, 0), 2);
	line(dstImg, sceneCorners[3], sceneCorners[0], Scalar(0, 255, 0), 2);

	Mat differenceImg;
	absdiff(srcImg, croppedDstImg, differenceImg);

	double n;
	n = norm(srcImg, croppedDstImg);
	printf("%f\n", n);

	//imshow("Matches: Src image (left) to dst (right)", matchImg);
	//imshow("Original", srcImg);
	//imshow("Matched", dstImg);
	//imshow("Cropped", croppedDstImg);
	//imshow("Aligned", alignedDstImg);
	imshow("Difference", differenceImg);

	waitKey(0);

	return 0;
}

