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

int SWS = 5;
int PFS = 5;
int preFiltCap = 29;
int minDisp = -25;
int numOfDisp = 128;
int TxtrThrshld = 100;
int unicRatio = 10;
int SpcklRng = 15;
int SpklWinSze = 100;

//string folder_name = "/Users/liubingxin/Documents/cse520s_project/zcyd";
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

Mat filtered_disp,solved_disp,solved_filtered_disp;
Rect ROI;

// check
Mat cameraMatrixL = (Mat_<double>(3, 3) << 5.717662471089529e+02, 0, 3.599894403338598e+02,\
                                            0, 7.636306316305219e+02, 2.947943548027684e+02,\
                                            0, 0, 1);
Mat cameraMatrixR = (Mat_<double>(3, 3) << 5.679819209188780e+02, 0, 3.452859895943097e+02,\
                                            0, 7.609519703287510e+02, 2.895787685416111e+02,\
                                            0, 0, 1);
Mat distCoeffL = (Mat_<double>(5, 1) << -0.284128304263526, 0.603976907145943,\
                         0.002650751621832, 6.311318302912327e-04, \
                         0.00000);
Mat distCoeffR = (Mat_<double>(5, 1) << -0.236473331555064, 0.319212543138987,\
                         0.002170239381114, -0.002597546865672,\
                         0.00000);
Mat T = (Mat_<double>(3, 1) << -60.513327359166034, -0.037255253186699, -2.189518833850237);
Mat R = (Mat_<double>(3, 3) << 1, -0.001208891549226, -0.002628955911107,\
                              0.001199053991384, 1, -0.003740364031551,\
                              0.002633457325662, 0.003737196112728, 1);

long long getTimestamp() {
    const std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    const std::chrono::microseconds epoch = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return  epoch.count();
}

void loadParams()
{
    fprintf(stderr, "Loading params...\n");
    string filename = folder_name + "/para/3dmap_set.yml";
    FileStorage fs;
    if (fs.open(filename, FileStorage::READ))
    {
        fs["SWS"] >> SWS;
        fs["PFS"] >> PFS;
        fs["preFiltCap"] >> preFiltCap;
        fs["minDisp"] >> minDisp;
        fs["numOfDisp"] >> numOfDisp;
        fs["TxtrThrshld"] >> TxtrThrshld;
        fs["unicRatio"] >> unicRatio;
        fs["SpcklRng"] >> SpcklRng;
        fs["SpklWinSze"] >> SpklWinSze;
    }

    fprintf(stderr, "pfs = %d\n", PFS);

}

void saveParams()
{
    fprintf(stderr, "Saving params...\n");
    string filename = folder_name + "/para/3dmap_set.yml";
    FileStorage fs(filename, FileStorage::WRITE);
    fs << "SWS" << SWS << "PFS" << PFS << "preFiltCap" << preFiltCap << "minDisp" << minDisp << "numOfDisp" << numOfDisp
       << "TxtrThrshld" << TxtrThrshld << "unicRatio" << unicRatio  << "SpcklRng" << SpcklRng << "SpklWinSze" << SpklWinSze;
}

void stereo_depth_map(Mat left, Mat right)
{
    Mat conf_map = Mat(left.rows, left.cols, CV_8U);
    conf_map = Scalar(255);
    
    Ptr<StereoBM> bm = StereoBM::create(16,9);

    auto wls_filter = ximgproc::createDisparityWLSFilter(bm);
    Ptr<StereoMatcher> right_matcher = ximgproc::createRightMatcher(bm);


    if (SWS < 5) SWS = 5;
    if (SWS %2 == 0) SWS += 1;
    if (SWS > left.rows) SWS = left.rows - 1;
    if (numOfDisp < 16) numOfDisp = 16;
    if (numOfDisp % 16 != 0) numOfDisp -= (numOfDisp %16);
    if (preFiltCap < 1) preFiltCap = 1;

    bm->setPreFilterCap(preFiltCap);
    bm->setBlockSize(SWS);
    bm->setMinDisparity(minDisp);
    bm->setNumDisparities(numOfDisp);
    bm->setTextureThreshold(TxtrThrshld);
    bm->setUniquenessRatio(unicRatio);
    bm->setSpeckleWindowSize(SpklWinSze);
    bm->setSpeckleRange(SpcklRng);
    bm->setDisp12MaxDiff(1);

    Mat l_disp, r_disp, disp8, colored;
    bm->compute(left, right, l_disp);
    right_matcher->compute(right, left, r_disp);

    wls_filter->setLambda(8000.0);
    wls_filter->setSigmaColor(1.5);
    wls_filter->filter(l_disp,left,filtered_disp,r_disp);

    conf_map = wls_filter->getConfidenceMap();
    ROI = wls_filter->getROI();

    Mat raw_disp_vis;
    ximgproc::getDisparityVis(l_disp,raw_disp_vis,15.0);
    //namedWindow("raw disparity", WINDOW_AUTOSIZE);
    //imshow("raw disparity", raw_disp_vis);
    Mat filtered_disp_vis;
    ximgproc::getDisparityVis(filtered_disp,filtered_disp_vis,15.0);
    //namedWindow("filtered disparity", WINDOW_AUTOSIZE);
    //imshow("filtered disparity", filtered_disp_vis);


    filtered_disp_vis.convertTo(disp8, CV_8U);
    applyColorMap(disp8, colored, COLORMAP_JET);
    imshow("Image", colored);

    Mat threshold_output;
    Mat copyImage = disp8.clone();
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    RNG rng(12345);
    threshold(copyImage, threshold_output, 20, 255, cv::THRESH_BINARY);
    findContours(threshold_output, contours, hierarchy, \
            cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, Point(0, 0));
    
    // calculate hull
    vector<vector<Point>> hull(contours.size());
    vector<vector<Point>> result;

    for (int i = 0; i < contours.size(); ++i)
        convexHull(Mat(contours[i]), hull[i], false);
    
    Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);

    for (int i = 0; i < contours.size(); ++i)
    {
        if (contourArea(contours[i]) < 500) continue;
        result.push_back(hull[i]);
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
        drawContours(drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());
    }
    imshow("contours", drawing);
}

void onTrackbar(int, void *)
{
    stereo_depth_map(rectifyImageL, rectifyImageR);
}

void onMinDisp(int, void *)
{
    minDisp -= 40;
    stereo_depth_map(rectifyImageL, rectifyImageR);
}


int main(int argc, char* argv[])
{
    string imageToDisp = "";
    int photo_width = 640;
    int photo_height = 480;
    int image_width = 320;
    int image_height = 240;

    Size imageSize(photo_width, photo_height);

    //imgLeft = imread( folder_name + "/photos/" + argv[1] + "L.jpg", IMREAD_GRAYSCALE );
    //imgRight = imread( folder_name + "/photos/" + argv[1] + "R.jpg", IMREAD_GRAYSCALE );
    //imshow("Left Image", imgLeft);
    //imshow("Right Image", imgRight);

    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize,\
                     R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY, 0, imageSize, \
                     &validROIL, &validROIR);
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1,\
                             mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1,\
                            mapRx, mapRy);

    loadParams();

    cout << "Load Parameters done.\n";
    /*
    namedWindow("Image");
    moveWindow("Image", 50, 100);
    namedWindow("Left");
    moveWindow("Left", 450, 100);
    namedWindow("Right");
    moveWindow("Right", 850, 100);

    createTrackbar("SWS", "Image", &SWS, 255, onTrackbar);
    createTrackbar("PFS", "Image", &PFS, 255, onTrackbar);
    createTrackbar("PreFiltCap", "Image", &preFiltCap, 63, onTrackbar);
    createTrackbar("MinDISP", "Image", &minDisp, 100, onMinDisp);
    createTrackbar("NumOfDisp", "Image", &numOfDisp, 256, onTrackbar);
    createTrackbar("TxtrThrshld", "Image", &TxtrThrshld, 100, onTrackbar);
    createTrackbar("UnicRatio", "Image", &unicRatio, 100, onTrackbar);
    createTrackbar("SpcklRng", "Image", &SpcklRng, 40, onTrackbar);
    createTrackbar("SpklWinSze", "Image", &SpklWinSze, 300, onTrackbar);
    */

    cout << "Creat Trackbar done.\n";

    long long total_time;
    long long start_time = getTimestamp();
    float avgFps = 0.0;
    int frame_number = 0;
    int prevFrameNumber = 0;

    long long remap_time_start;
    long long remap_time_end;
    long long disp_time_start;
    long long disp_time_end;
    long long frame_time_start;
    long long frame_time_end;


    //rectifyImageL = imgLeft;
    //rectifyImageR = imgRight;

    cout << "Remap done.\n";

    VideoCapture cam0("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    VideoCapture cam1("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
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

    start_time = getTimestamp();

    while (1)
    {
        frame_time_start = getTimestamp();
        cam0 >> imgRight;
        cam1 >> imgLeft;

        imshow("Left Image", imgLeft);

        
        cvtColor(imgLeft, grayImageL, COLOR_RGB2GRAY);
        cvtColor(imgRight, grayImageR, COLOR_RGB2GRAY);
        //cout << "cvtColor done.\n";

        //remap_time_start = getTimestamp();
        remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
        remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);
        //remap_time_end = getTimestamp();
        //cout << remap_time_end << endl;
        //cout << "remap time: " << (remap_time_end - remap_time_start) / 1000000.0 << endl;

        //disp_time_start = getTimestamp();
        stereo_depth_map(rectifyImageL, rectifyImageR);
        //disp_time_end = getTimestamp();
        //cout << "disp time: " << (disp_time_end - disp_time_start) / 1000000.0 << endl;

        frame_number++;

        //cv::imshow("Left", rectifyImageL);
        //cv::imshow("Right", rectifyImageR);
    
        char k = waitKey(1);
        if( k == 's' || k == 'S')
        {
            saveParams();
            break;
        } else if (k == 'q' || k == 'Q') {
            cout << "Quit the program." << endl;
            break;
        }

        frame_time_end = getTimestamp();
        cout << "frame time " << (frame_time_end - frame_time_start) / 1000000.0 << endl;
    }

    total_time = getTimestamp();
    float fps = frame_number / ((total_time - start_time) / 1000000);
    fprintf(stdout, "Frames number: %lld\nTotal time: %lld\nAverage FPS: %f\n", frame_number, total_time - start_time, fps);
    cam0.release();
    cam1.release();
    destroyAllWindows();
    return 0;

}


        