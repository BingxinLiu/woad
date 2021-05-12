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
#include <opencv2/cudastereo.hpp>


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

void threadfn();

void stereo_depth_map(Mat left, Mat right);

int main(int argc, char* argv[])
{

    cout << "OpenCV version : " << CV_VERSION << endl;

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
        video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)2/1 !\
        nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    VideoCapture cam1("nvarguscamerasrc sensor-id=1 ! \
        video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)2/1 !\
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

    for(int i = 1; i <= 4; ++i)
    {
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

        if(i == 4) i = 0;
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
    numDisparities= 64;
    P1 = 8 * 3 * window_size;
    P2 = 32 * 3 * window_size;
    disp12MaxDiff = 12;
    uniquenessRatio = 10;
    speckleWindowSize = 50;
    speckleRange = 32;
    preFilterCap = 63;
    //mode = cv::StereoSGM::MODE_SGBM_3WAY;
    mode = 3;

    Ptr<cuda::StereoSGM> sgm = cuda::createStereoSGM(minDisparity, numDisparities);
    sgm->setBlockSize(blockSize);
    sgm->setDisp12MaxDiff(disp12MaxDiff);
    sgm->setMinDisparity(minDisparity);
    sgm->setNumDisparities(numDisparities);
    sgm->setP1(P1);
    sgm->setP2(P2);
    sgm->setUniquenessRatio(uniquenessRatio);
    sgm->setMode(mode);
    sgm->setSpeckleRange(speckleRange);
    sgm->setPreFilterCap(preFilterCap);

    Ptr<cuda::StereoSGM> right_matcher = \
        cv::ximgproc::createRightMatcher(sgm);
        //cuda::createStereoSGM(minDisparity, numDisparities, P1, P2, uniquenessRatio, mode);
    
    int lmbda = 80000;
    float sigma = 1.3;
    int visual_multiplier;

    Ptr<ximgproc::DisparityWLSFilter> wls_filter =\
        ximgproc::createDisparityWLSFilter(sgm);
    wls_filter->setLambda(lmbda);
    wls_filter->setSigmaColor(sigma);

    Mat l_disp, r_disp, filtered_disp;
    cuda::GpuMat gl_disp, gr_disp, g_left, g_right;

    g_left.upload(left);
    g_right.upload(right);

    sgm->compute(g_left, g_right, gl_disp);
    right_matcher->compute(g_right, g_left, gr_disp);

    gl_disp.download(l_disp);
    gr_disp.download(r_disp);

    wls_filter->filter(l_disp, left, filtered_disp, r_disp);
    imshow("Filtered sgm:", filtered_disp);

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
    imshow("left disp", l_disp);
    imshow("right disp", r_disp);
    imshow("DisplayMat:", displayMat);
    //cout << filtered_disp << endl;
    //cout << normalized << endl;


}

