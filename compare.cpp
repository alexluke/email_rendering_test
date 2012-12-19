#include <cv.h>
#include <highgui.h>

using namespace cv;

#define SECTION_THRESHOLD 10
#define DEBUG 1
#ifdef DEBUG
#define PRINT_RECT(rect) printf("Rect at (%d, %d) %dx%d\n", rect.x, rect.y, rect.width, rect.height)
#else
#define PRINT_RECT(rect) do {} while(0)
#endif

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

double matchImage(Mat target, Mat toMatch, Rect region=Rect()) {
	if (region.width == 0 || region.height == 0)
		region = Rect(0, 0, toMatch.cols, toMatch.rows);

	Mat section = toMatch(region);

#ifdef DEBUG
	imshow("Original", toMatch);
#endif

	SurfFeatureDetector detector(2000);
	vector<KeyPoint> srcFeatures, dstFeatures;

	detector.detect(section, srcFeatures);
	detector.detect(target, dstFeatures);

	SurfDescriptorExtractor extractor;
	Mat srcDescriptors, dstDescriptors;
	extractor.compute(section, srcFeatures, srcDescriptors);
	extractor.compute(target, dstFeatures, dstDescriptors);

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
	H = findHomography(src2DFeatures, dst2DFeatures, RANSAC, 3, outlierMask);

#ifdef DEBUG
	Mat matchImg;
	drawMatches(section, srcFeatures, target, dstFeatures, matches, matchImg, Scalar::all(-1), Scalar::all(-1),
		reinterpret_cast<const vector<char>&>(outlierMask));
	imshow("Matches: Src image (left) to dst (right)", matchImg);
#endif

	vector<Point2f> objCorners(4);
	objCorners[0] = Point(0, 0);
	objCorners[1] = Point(section.cols, 0);
	objCorners[2] = Point(section.cols, section.rows);
	objCorners[3] = Point(0, section.rows);

	vector<Point2f> sceneCorners(4);
	perspectiveTransform(objCorners, sceneCorners, H);

#ifdef DEBUG
	Mat originalTarget;
	target.copyTo(originalTarget);


	line(originalTarget, sceneCorners[0], sceneCorners[1], Scalar(0, 255, 0), 2);
	line(originalTarget, sceneCorners[1], sceneCorners[2], Scalar(0, 255, 0), 2);
	line(originalTarget, sceneCorners[2], sceneCorners[3], Scalar(0, 255, 0), 2);
	line(originalTarget, sceneCorners[3], sceneCorners[0], Scalar(0, 255, 0), 2);

	namedWindow("Matched", CV_WINDOW_NORMAL);
	imshow("Matched", originalTarget);
	waitKey(0);
#endif

	int top, left, bottom, right;
	top = min(sceneCorners[0].y, sceneCorners[1].y);
	bottom = max(sceneCorners[2].y, sceneCorners[3].y);
	left = min(sceneCorners[0].x, sceneCorners[3].x);
	right = max(sceneCorners[1].x, sceneCorners[2].x);
	Rect dstRect = Rect(left, top, right - left, bottom - top);

	Point p = dstRect.tl() - Point(sceneCorners[0].x, sceneCorners[0].y);
	region.x += p.x;
	region.y += p.y;
	region.width = dstRect.width;
	region.height = dstRect.height;

	section = toMatch(region);

	target = target(dstRect);

#ifdef DEBUG
	rectangle(originalTarget, dstRect, Scalar(255, 0, 0), 2);

	imshow("Cropped", target);
	imshow("Matched", originalTarget);
	waitKey(0);
#endif

#ifdef DEBUG
	Mat differenceImg;
	absdiff(section, target, differenceImg);

	imshow("Difference", differenceImg);
#endif

	return norm(section, target);
}

vector<Rect> getSections(Mat img, Rect region=Rect()) {
	if (region.width == 0 || region.height == 0)
		region = Rect(0, 0, img.cols, img.rows);

	vector<Rect> sections;
	Mat tmpImg = img(region);
	cvtColor(tmpImg, tmpImg, CV_BGR2GRAY);
	threshold(tmpImg, tmpImg, 150, 255, THRESH_BINARY);

	int consecutiveBlankRows = 0,
		startContent = 0,
		endContent = 0;
	for (int row = 1; row < tmpImg.rows - 1; row++) {
		unsigned char* p = tmpImg.ptr<unsigned char>(row);
		bool blankRow = true;
		for (int col = 1; col < tmpImg.cols - 1; col++) {
			p++;
			if (*p == 0) {
				blankRow = false;
				break;
			}
		}
		if (blankRow) {
			consecutiveBlankRows++;
		} else {
			if (consecutiveBlankRows >= SECTION_THRESHOLD) {
				Rect r = Rect(0, startContent, tmpImg.cols, endContent - startContent);
				r += region.tl();
				sections.push_back(r);
				startContent = endContent = row - 1;
			} else {
				endContent = row + 2;
			}
			consecutiveBlankRows = 0;
		}
	}
	Rect r = Rect(0, startContent, tmpImg.cols, endContent - startContent);
	r += region.tl();
	sections.push_back(r);

	return sections;
}

int main(int argc, char** argv) {
	Mat srcImg, dstImg;
	srcImg = imread("images/email1/source.png");
	dstImg = imread("images/email1/gmail.png");

	if (!(srcImg.data && dstImg.data)) {
		printf("No image data\n");
		return -1;
	}

	Rect srcRect = getBorder(srcImg);

	double matchValue = matchImage(dstImg, srcImg, srcRect);
	printf("Total match value: %f\n", matchValue);

	vector<Rect> sections = getSections(srcImg, srcRect);
	char name[50];
	for (unsigned int i = 0; i < sections.size(); i++) {
#ifdef DEBUG
		PRINT_RECT(sections[i]);
		Mat t = srcImg(sections[i]);
		sprintf(name, "Section %d", i);
		imshow(name, t);
#endif
		matchValue = matchImage(dstImg, srcImg, sections[i]);
		printf("Section #%d match value: %f\n", i, matchValue);
	}
	waitKey(0);

	return 0;
}

