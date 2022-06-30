#include <iostream>
#include <string>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {


    Mat img_1 = imread(argv[1], IMREAD_GRAYSCALE);
    Mat img_2= imread(argv[2], IMREAD_GRAYSCALE);

    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1_freak, descriptors_2_freak, descriptors_1_orb, descriptors_2_orb;
    std::vector<DMatch> matches;

    // Detector
    Ptr<FastFeatureDetector> detector;
    detector = FastFeatureDetector::create();

    // Descriptor
    Ptr<xfeatures2d::FREAK> extractor_freak = xfeatures2d::FREAK::create();
    Ptr<ORB> extractor_orb = ORB::create();

    // Matching
    BFMatcher matcher(NORM_HAMMING);

    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);
    cout << "Keypoints detected in Image 1: " << keypoints_1.size() << endl;
    cout << "Keypoints detected in Image 2: " << keypoints_2.size() << endl;

    // ORB descriptor time
    chrono::steady_clock::time_point orb_start = chrono::steady_clock::now();
    extractor_orb->compute(img_1, keypoints_1, descriptors_1_orb);
    extractor_orb->compute(img_2, keypoints_2, descriptors_2_freak);
    chrono::steady_clock::time_point orb_end = chrono::steady_clock::now();
    double t_orb = chrono::duration_cast<chrono::duration<double, milli>>(orb_end - orb_start).count();
    cout << "ORB descriptor time: " << t_orb << " milliseconds" << endl;

    // FREAK descriptor time
    chrono::steady_clock::time_point freak_start = chrono::steady_clock::now();
    extractor_freak->compute(img_1, keypoints_1, descriptors_1_freak);
    extractor_freak->compute(img_2, keypoints_2, descriptors_2_freak);
    chrono::steady_clock::time_point freak_end = chrono::steady_clock::now();
    double t_freak = chrono::duration_cast<chrono::duration<double, milli>>(freak_end - freak_start).count();
    cout << "FREAK descriptor time: " << t_freak << " milliseconds" << endl;

    matcher.match(descriptors_1_freak, descriptors_2_freak, matches);


    // Display Results
    Mat matchingResult;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, matchingResult);
    imshow("matches", matchingResult);
    waitKey(0);

    return 0;
}