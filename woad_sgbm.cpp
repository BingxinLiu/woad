#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<string>
#include<iostream>
#include<opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include<opencv2/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/calib3d.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/core/utility.hpp>
#include<opencv2/ximgproc.hpp>
#include<sys/time.h>


using namespace cv;
using namespace std;

string folder_name = "/home/lbx/cse520s/zcyd";
string calibration_data_folder = "";

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL, validROIR;

Mat imgLeft, imgRight;
Mat mapLx, mapLy, mapRx, mapRy;
Mat Rl, Rr, Pl, Pr, Q;
Mat xyz;

int window_size, minDisparity, numDisparities, blockSize,\
    P1, P2, disp12MaxDiff, uniquenessRatio, speckleWindowSize,\
    speckleRange, preFilterCap, mode;

Mat cameraMatrixL = (Mat_<double>(3, 3) << \
    5.717662471089529e+02, 0, 3.599894403338598e+02,\
    0, 7.636306316305219e+02, 2.947943548027684e+02,\
    0, 0, 1);
Mat cameraMatrixR = (Mat_<double>(3, 3) << \
    5.679819209188780e+02, 0, 3.452859895943097e+02,\
    0, 7.609519703287510e+02, 2.895787685416111e+02,\
    0, 0, 1);
Mat distCoeffL = (Mat_<double>(5, 1) << \
    -0.284128304263526, 0.603976907145943,\
    0.002650751621832, 6.311318302912327e-04, 0.00000);
Mat distCoeffR = (Mat_<double>(5, 1) << \
    -0.236473331555064, 0.319212543138987,\
    0.002170239381114, -0.002597546865672, 0.00000);
Mat T = (Mat_<double>(3, 1) << \
    -60.513327359166034, -0.037255253186699, -2.189518833850237);
Mat R = (Mat_<double>(3, 3) << \
    1, -0.001208891549226, -0.002628955911107,\
    0.001199053991384, 1, -0.003740364031551,\
    0.002633457325662, 0.003737196112728, 1);

long long getTimestamp();
void stereo_depth_map(Mat left, Mat right);
void calculateDistance(Mat disp);

int main(int argc, char* argv[])
{
    int photo_width = 640;
    int photo_height = 480;
    int image_width = 640;
    int image_height = 480;

    Size imageSize(photo_width, photo_height);

    stereoRectify(cameraMatrixL, distCoeffL,\
    cameraMatrixR, distCoeffR, imageSize,\
    R, T, Rl, Rr, Pl, Pr, Q,\
    CALIB_ZERO_DISPARITY, 0, imageSize, \
    &validROIL, &validROIR);
    initUndistortRectifyMap(cameraMatrixL, distCoeffL,\
    Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR,\
    Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

    VideoCapture cam0("nvarguscamerasrc sensor-id=0 ! \
        video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)1/1 !\
        nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    VideoCapture cam1("nvarguscamerasrc sensor-id=1 ! \
        video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)1/1 !\
        nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);

    if(!cam0.isOpened())
    {
       printf("cam0 is not opened.\n");
       return -1;
    }
    if(!cam1.isOpened())
    {
       printf("cam1 is not opened.\n");
       return -1;
    }

    long long frame_time_start;
    long long frame_time_end;

    while (1)
    {
        frame_time_start = getTimestamp();
        cam0 >> imgRight;
        cam1 >> imgLeft;

        remap(imgLeft, rectifyImageL, mapLx, mapLy,\
            INTER_LINEAR, BORDER_CONSTANT);
        remap(imgRight, rectifyImageR, mapRx, mapRy,\
            INTER_LINEAR, BORDER_CONSTANT);

        cvtColor(rectifyImageL, grayImageL, COLOR_BGR2GRAY);
        cvtColor(rectifyImageR, grayImageR, COLOR_BGR2GRAY);

        stereo_depth_map(grayImageL, grayImageR);

        char k = waitKey(1);
        if (k == 'q' || k == 'Q') {
            cout << "Quit the program." << endl;
            break;
        }

        frame_time_end = getTimestamp();
        cout << "frame time " << (frame_time_end - frame_time_start) / 1000000.0 << endl;
    }
    cam0.release();
    cam1.release();
    destroyAllWindows();
    return 0;
}


void stereo_depth_map(Mat left, Mat right)
{
    window_size = blockSize = 3;
    minDisparity = -1;
    numDisparities= 5*16;
    P1 = 8 * 3 * window_size;
    P2 = 32 * 3 * window_size;
    disp12MaxDiff = 12;
    uniquenessRatio = 10;
    speckleWindowSize = 50;
    speckleRange = 32;
    preFilterCap = 63;
    mode = cv::StereoSGBM::MODE_SGBM_3WAY;

    Ptr<StereoSGBM> sgbm = StereoSGBM::create(numDisparities, blockSize);
    sgbm->setBlockSize(blockSize);
    sgbm->setDisp12MaxDiff(disp12MaxDiff);
    sgbm->setMinDisparity(minDisparity);
    sgbm->setMode(mode);
    sgbm->setNumDisparities(numDisparities);
    sgbm->setP1(P1);
    sgbm->setP2(P2);
    sgbm->setPreFilterCap(preFilterCap);
    sgbm->setSpeckleRange(speckleRange);
    sgbm->setUniquenessRatio(uniquenessRatio);

    Ptr<StereoMatcher> right_matcher = \
        cv::ximgproc::createRightMatcher(sgbm);
    
    int lmbda = 80000;
    float sigma = 1.3;
    int visual_multiplier;

    Ptr<ximgproc::DisparityWLSFilter> wls_filter =\
        ximgproc::createDisparityWLSFilter(sgbm);
    wls_filter->setLambda(lmbda);
    wls_filter->setSigmaColor(sigma);

    Mat l_disp, r_disp, filtered_disp;
    sgbm->compute(left, right, l_disp);
    right_matcher->compute(right, left, r_disp);

    wls_filter->filter(l_disp, left, filtered_disp, r_disp);
    imshow("Filtered sgbm:", filtered_disp);

    /*
    Mat conf_map = Mat(left.rows, left.cols, CV_8U);
    conf_map = Scalar(255);
    conf_map = wls_filter->getConfidenceMap();
    Rect ROI = wls_filter->getROI();
    Mat raw_disp_vis;
    ximgproc::getDisparityVis(l_disp,raw_disp_vis);
    namedWindow("raw disparity", WINDOW_AUTOSIZE);
    imshow("raw disparity", raw_disp_vis);
    Mat filtered_disp_vis;
    ximgproc::getDisparityVis(filtered_disp,filtered_disp_vis);
    namedWindow("filtered disparity", WINDOW_AUTOSIZE);
    imshow("filtered disparity", filtered_disp_vis);
    */

    Mat normalized;
    normalize(filtered_disp, normalized, 255, 0, NORM_MINMAX, -1, noArray());
    
    Mat displayMat;
    normalized.convertTo(displayMat, CV_8U);

    imshow("left", left);
    //imshow("right", right);
    //imshow("left disp", l_disp);
    //imshow("right disp", r_disp);

    
    imshow("DisplayMat:", displayMat);

    //cout << filtered_disp << endl;
    //cout << normalized << endl;

    calculateDistance(displayMat);

    imshow("Depth map:", (2.6 * 60) / displayMat);


}

void calculateDistance(Mat disp)
{
    Mat mat = disp.clone();
    if (mat.channels() > 1)
    {
        cout << "Error: Mat channel is: " << mat.channels() << endl;
        return;
    } 
    int width = mat.size().width;
    int height = mat.size().height;

    //cout << mat.colRange(120, 130) << endl;

    uchar _result[width]; 

    //transpose(mat, mat);
    
    for (int i = 0; i < width; ++i)
    {
        double minVal, maxVal;
        minMaxIdx(mat.col(i), NULL, &maxVal, NULL, NULL);
        _result[i] = (uchar) maxVal;

    }
    //cout << _result[120] << " " << _result[121] << " " << _result[122] << endl;

    Mat resultMat = Mat(1, width, CV_8U);
    resultMat.data = _result;
    //cout << resultMat.colRange(120, 130) << endl;
    resize(resultMat, resultMat, Size(width, 20),0, 0, 1);

    cout << "result::r=" << resultMat.size().height\
        << "result::c" << resultMat.size().width << endl;

    applyColorMap(resultMat, resultMat, COLORMAP_SUMMER);
    medianBlur ( resultMat, resultMat, 5 );
    //imshow("AFter medianBlur", displayMat);
    namedWindow("Distance", WINDOW_AUTOSIZE);
    imshow("Distance", resultMat);
    //waitKey(0);

}

long long getTimestamp() {
    const std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    const std::chrono::microseconds epoch = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return  epoch.count();
}
