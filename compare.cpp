#include <cv.h>
#include <highgui.h>

using namespace cv;

int main(int argc, char** argv) {
	Mat srcImg, dstImg;
	srcImg = imread("images/email1/source.png");
	dstImg = imread("images/email1/gmail.png");

	if (!(srcImg.data && dstImg.data)) {
		printf("No image data\n");
		return -1;
	}

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

	Mat differenceImg;
	absdiff(srcImg, alignedDstImg, differenceImg);

	double n;
	n = norm(srcImg, alignedDstImg);
	printf("%f\n", n);

	imshow("Matches: Src image (left) to dst (right)", matchImg);
	imshow("Original", srcImg);
	imshow("Aligned", alignedDstImg);
	imshow("Difference", differenceImg);

	waitKey(0);

	return 0;
}

