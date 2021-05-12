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

const int MODE = 1;
const bool DEBUG = true;
const bool GRAY = false;
//#define BIG
#define FILTER

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

// initial Matrix needed by rectification

//Mat cameraMatrixL, cameraMatrixR, distCoeffL, distCoeffR, T, R;
#ifdef BIG
Mat cameraMatrixL = (Mat_<double>(3, 3) << \
    1.134793835292295e+03, 0, 6.905237450586839e+02,\
    0, 1.133294765139664e+03, 4.085216204500437e+02,\
    0, 0, 1);
Mat cameraMatrixR = (Mat_<double>(3, 3) << \
    1.131741524489375e+03, 0, 6.649315301894233e+02,\
    0, 1.132811113305317e+03, 4.012411967470943e+02,\
    0, 0, 1);
Mat distCoeffL = (Mat_<double>(5, 1) << \
    -0.230070637746889, 0.818218200600011,\
    0.001956239154185, 0.001118575422353, 0.00000);
Mat distCoeffR = (Mat_<double>(5, 1) << \
    -0.221235491366580, 0.613761338473414,\
    5.969755239050328e-04, 0.001339915858426, 0.00000);
Mat T = (Mat_<double>(3, 1) << \
    -60.304073993534490, 0.365811910962598, -0.765052029245452);
Mat R = (Mat_<double>(3, 3) << \
    1, 2.488767718491064e-04, -0.005869730264495,\
    -2.778898279817759e-04, 1, -0.004942528615310,\
    0.0058684282622352, 0.004944074455557, 1);
#else
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
#endif



void stereo_depth_map(Mat left, Mat right);
long long getTimestamp();
void loadParams();
void saveParams();
void calculateDistance(Mat disp);

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
#ifdef BIG
    int photo_width = 1280;
    int photo_height = 720;
    int image_width = 1280;
    int image_height = 720;
#else
    int photo_width = 640;
    int photo_height = 480;
    int image_width = 640;
    int image_height = 480;
#endif

    Size imageSize(photo_width, photo_height);

    //imgLeft = imread( folder_name + "/photos/" + argv[1] + "L.jpg", IMREAD_GRAYSCALE );
    //imgRight = imread( folder_name + "/photos/" + argv[1] + "R.jpg", IMREAD_GRAYSCALE );
    //imshow("Left Image", imgLeft);
    //imshow("Right Image", imgRight);

    stereoRectify(cameraMatrixL, distCoeffL,\
        cameraMatrixR, distCoeffR, imageSize,\
        R, T, Rl, Rr, Pl, Pr, Q,\
        CALIB_ZERO_DISPARITY, 0, imageSize, \
        &validROIL, &validROIR);
    initUndistortRectifyMap(cameraMatrixL, distCoeffL,\
        Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR,\
        Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

    loadParams();

    cout << "Load Parameters done.\n";
    
    namedWindow("Image");
    moveWindow("Image", 50, 100);
    namedWindow("Left");
    moveWindow("Left", 450, 100);
    namedWindow("Right");
    moveWindow("Right", 850, 100);


    createTrackbar("SWS", "Image", &SWS, 255, onTrackbar);
    createTrackbar("PFS", "Image", &PFS, 255, onTrackbar);
    createTrackbar("PreFiltCap", "Image", &preFiltCap, 63, onTrackbar);
    createTrackbar("MinDISP", "Image", &minDisp, 100, onTrackbar);
    createTrackbar("NumOfDisp", "Image", &numOfDisp, 256, onTrackbar);
    createTrackbar("TxtrThrshld", "Image", &TxtrThrshld, 100, onTrackbar);
    createTrackbar("UnicRatio", "Image", &unicRatio, 100, onTrackbar);
    createTrackbar("SpcklRng", "Image", &SpcklRng, 40, onTrackbar);
    createTrackbar("SpklWinSze", "Image", &SpklWinSze, 300, onTrackbar);


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

#ifdef BIG
    VideoCapture cam0("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=1280, height=720, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    VideoCapture cam1("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=1280, height=720, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
#else
    VideoCapture cam0("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    VideoCapture cam1("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
#endif

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

    if (DEBUG) cout << "Starting Loop" << endl;
    while (1)
    {
        frame_time_start = getTimestamp();
        cam0 >> imgRight;
        cam1 >> imgLeft;

        imshow("Left Image", imgLeft);

        if (GRAY)
        {
            grayImageL = imgLeft;
            grayImageR = imgRight;
        } else 
        {
            cvtColor(imgLeft, grayImageL, COLOR_BGR2GRAY);
            cvtColor(imgRight, grayImageR, COLOR_BGR2GRAY);
        }

        if (DEBUG) cout << "cvtColor done" << endl;

        //remap_time_start = getTimestamp();
        remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR, BORDER_CONSTANT);
        remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR, BORDER_CONSTANT);
        //remap_time_end = getTimestamp();
        //cout << remap_time_end << endl;
        //cout << "remap time: " << (remap_time_end - remap_time_start) / 1000000.0 << endl;
        //equalizeHist( rectifyImageL, rectifyImageL );
        //equalizeHist( rectifyImageR, rectifyImageR );
        if (DEBUG) cout << "remap done" << endl;

        //disp_time_start = getTimestamp();
        stereo_depth_map(rectifyImageL, rectifyImageR);
        //disp_time_end = getTimestamp();
        //cout << "disp time: " << (disp_time_end - disp_time_start) / 1000000.0 << endl;

        if (DEBUG) cout << "stereo_depth_map done" << endl;
        frame_number++;

        cv::imshow("Left", rectifyImageL);
        cv::imshow("Right", rectifyImageR);
    
        if (DEBUG) cout << "rectify image show done" << endl;
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
    fprintf(stdout, "Frames number: %d\nTotal time: %lld\nAverage FPS: %f\n", frame_number, total_time - start_time, fps);
    cam0.release();
    cam1.release();
    destroyAllWindows();
    return 0;

}

void stereo_depth_map(Mat left, Mat right)
{
    Ptr<StereoBM> bm = nullptr;
    Ptr<cuda::StereoBM> cudabm = nullptr;
    cuda::GpuMat g_left, g_right, gl_disp, gr_disp;
#ifdef FILTER
    Ptr<StereoMatcher> right_matcher = nullptr;
    //Ptr<StereoMatcher> cuda_right_matcher = nullptr;
    Ptr<cuda::StereoBM> cuda_right_matcher = nullptr;
    Ptr<ximgproc::DisparityWLSFilter> wls_filter = nullptr;
    Ptr<ximgproc::DisparityWLSFilter> cuda_wls_filter = nullptr;
    Mat filtered_disp,solved_disp,solved_filtered_disp;
    Rect ROI;
#endif

    if (SWS < 5) SWS = 5;
    if (SWS %2 == 0) SWS += 1;
    if (SWS > left.rows) SWS = left.rows - 1;
    if (numOfDisp < 16) numOfDisp = 16;
    if (PFS % 2 == 0) PFS += 1;
    if (numOfDisp % 16 != 0) numOfDisp -= (numOfDisp %16);
    if (preFiltCap < 1) preFiltCap = 1;
    Mat conf_map = Mat(left.rows, left.cols, CV_8U);
    conf_map = Scalar(255);

    if (DEBUG) cout << "Initialize done" << endl;

    if (MODE == 0 || MODE == 2) 
    {
        bm = StereoBM::create(numOfDisp, SWS);
        bm->setPreFilterType(StereoBM::PREFILTER_XSOBEL);
        bm->setPreFilterSize(PFS);
        bm->setPreFilterCap(preFiltCap);
        bm->setBlockSize(SWS);
        bm->setMinDisparity(minDisp);
        bm->setNumDisparities(numOfDisp);
        bm->setTextureThreshold(TxtrThrshld);
        bm->setUniquenessRatio(unicRatio);
        bm->setSpeckleWindowSize(SpklWinSze);
        bm->setSpeckleRange(SpcklRng);
        bm->setDisp12MaxDiff(-1);
        bm->setROI1(validROIL);
        bm->setROI2(validROIR);
        if (DEBUG) cout << "Initialize bm arguments done" << endl;
    }
    if (MODE == 1 || MODE == 2)
    {
        cudabm = cuda::createStereoBM(numOfDisp, SWS);
        cudabm->setPreFilterType(0);
        cudabm->setPreFilterSize(PFS);
        cudabm->setPreFilterCap(preFiltCap);
        cudabm->setBlockSize(SWS);
        cudabm->setMinDisparity(minDisp + numOfDisp);
        cudabm->setNumDisparities(numOfDisp);
        cudabm->setTextureThreshold(TxtrThrshld);
        cudabm->setUniquenessRatio(unicRatio);
        cudabm->setSpeckleWindowSize(SpklWinSze);
        cudabm->setSpeckleRange(SpcklRng);
        cudabm->setDisp12MaxDiff(-1);
        cudabm->setROI1(validROIL);
        cudabm->setROI2(validROIR);
        if (DEBUG) {
            cout << "Initialize cudabm arguments done" << endl;
            cout << "Unique Ration:" << unicRatio << endl;
        } 
    }

    if (DEBUG) cout << "Initialize arguments done" << endl;

#ifdef FILTER
    cuda_wls_filter = ximgproc::createDisparityWLSFilter(cudabm);
    //cuda_right_matcher = ximgproc::createRightMatcher(cudabm);
    cuda_right_matcher = cuda::createStereoBM(numOfDisp, SWS);
    cuda_right_matcher->setMinDisparity(1 - numOfDisp);
    cuda_right_matcher->setPreFilterType(0);
    cuda_right_matcher->setPreFilterSize(PFS);
    cuda_right_matcher->setPreFilterCap(preFiltCap);
    cuda_right_matcher->setBlockSize(SWS);
    cuda_right_matcher->setNumDisparities(numOfDisp);
    cuda_right_matcher->setTextureThreshold(TxtrThrshld);
    cuda_right_matcher->setUniquenessRatio(unicRatio);
    cuda_right_matcher->setSpeckleWindowSize(SpklWinSze);
    cuda_right_matcher->setSpeckleRange(SpcklRng);
    cuda_right_matcher->setDisp12MaxDiff(-1);
    cuda_right_matcher->setROI1(validROIR);
    cuda_right_matcher->setROI2(validROIL);
#endif

    Mat l_disp, r_disp, disp8, colored;
    Mat lg_disp, g_disp8, o_disp;

    if (MODE == 1 || MODE == 2)
    {
        if (DEBUG) cout << "cudabm compute start" << endl;
        g_left.upload(left);
        g_right.upload(right);

        cudabm->compute(g_left, g_right, gl_disp);
        gl_disp.download(l_disp);
        imshow("Original left disp", l_disp);
        if (DEBUG) cout << "cudabm left compute done" << endl;

#ifdef FILTER
        if (DEBUG) cout << "??" << endl;
        cuda_right_matcher->compute(g_right, g_left, gr_disp);
        if (DEBUG) cout << "?" << endl;
        
        gr_disp.download(r_disp);
        if (DEBUG) cout << "cudabm right compute done" << endl;
        imshow("Original right disp", r_disp);
#endif

        //cuda::drawColorDisp(g_disp, g_disp, numOfDisp);

        //g_disp.download(lg_disp);
        // Mat diff_channels[4];
        // split(lg_disp, diff_channels);
        // Mat a = diff_channels[3];
        // if (DEBUG)
        // {
        //     auto size = a.size();
        //     cout << "rows:" << size.height << "\tcols:" << size.width << endl;
        // }
        // applyColorMap(a, a, COLORMAP_SUMMER);
        // imshow("alpha:", a);
        //imshow("Before cvt", lg_disp);
        //cvtColor(lg_disp, lg_disp, COLOR_RGBA2RGB);
        //imshow("After cvt", lg_disp);
        //if (DEBUG) cout << "lg_disp shape:" << lg_disp.channels() << endl;
        if (DEBUG) cout << "cudabm compute done" << endl;
    }
    if (MODE == 0 || MODE == 2)
    {
        bm->compute(left, right, l_disp);
        if (DEBUG) cout << "bm compute done" << endl;
    }

    if (DEBUG) cout << "compute done" << endl;

#ifdef FILTER
    if(DEBUG) cout << "right_matcher compute done" << endl;
    cuda_wls_filter->setLambda(5000.0);
    if(DEBUG) cout << "-?" << endl;
    cuda_wls_filter->setSigmaColor(1.5);
    if(DEBUG) cout << "?" << endl;
    cuda_wls_filter->filter(l_disp, left, filtered_disp,r_disp);
    imshow("Filtered disp:", filtered_disp);
    applyColorMap(filtered_disp, colored, COLORMAP_JET);
    imshow("After colorize:", colored);


    if(DEBUG) cout << "do filer done" << endl;
    //waitKey(0);
    conf_map = cuda_wls_filter->getConfidenceMap();
    ROI = cuda_wls_filter->getROI();
    
    Mat raw_disp_vis;
    //lg_disp_filter.convertTo(lg_disp_filter, CV_16S);
    /*
    if(DEBUG)
    {
        cout << "empty?" << lg_disp_filter.empty() << endl;
        cout << "depth?" << lg_disp_filter.depth() << endl;
        cout << "channels?" << lg_disp_filter.channels() << endl;

    }
    */
   /*
    ximgproc::getDisparityVis(l_disp,raw_disp_vis,1.0);
    namedWindow("raw disparity", WINDOW_AUTOSIZE);
    imshow("raw disparity", raw_disp_vis);
        //if (DEBUG) waitKey(0);
    Mat filtered_disp_vis;
    //filtered_disp.convertTo(filtered_disp, CV_16S);
    ximgproc::getDisparityVis(filtered_disp,filtered_disp_vis);
        //if (DEBUG) waitKey(0);
    namedWindow("filtered disparity", WINDOW_AUTOSIZE);
    imshow("filtered disparity", filtered_disp_vis);
    //filtered_disp_vis.convertTo(disp8, CV_8U);
    //applyColorMap(disp8, colored, COLORMAP_JET);
    //imshow("After filer:", filtered_disp_vis);
    */
#endif

    if (MODE == 0 || MODE == 2)
    {
        l_disp.convertTo(disp8, CV_8U);
        applyColorMap(disp8, colored, COLORMAP_JET);
        imshow("bm: disp", colored);
    }
    if (MODE == 1 || MODE == 2)
    {
        Mat displayMat;
        //double minVal, maxVal;
        //minMaxLoc(lg_disp, &minVal, &maxVal);
        //lg_disp = 255.*(lg_disp - minVal)/(maxVal - minVal);
        //lg_disp.convertTo(displayMat, CV_8U);

        //applyColorMap(l_disp, displayMat, COLORMAP_JET);
        //imshow("cudabm: disp", displayMat);
        //medianBlur ( displayMat, displayMat, 5 );
        //imshow("AFter medianBlur", displayMat);

        calculateDistance(filtered_disp);

        if (DEBUG) cout << "calculateDistance done" << endl;



        //fill hole
        //threshold(displayMat, displayMat, 220, 255, THRESH_BINARY_INV);
        //floodFill(displayMat, cv::Point(0,0), Scalar(255));
        //imshow("After floodFill", displayMat);
        //bitwise_not(displayMat, displayMat);
        
    }
    
    //l_disp.convertTo(disp8, CV_8U);
    //applyColorMap(disp8, colored, COLORMAP_JET);

    /*
    
    */
    //applyColorMap(displayMat, displayMat, COLORMAP_JET);



    /*
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
    */
}



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
    imshow("o_disp", mat);

    float _result[20][width]; 

    transpose(mat, mat);

    if(DEBUG) 
    {
        cout << "r:" << mat.size().height << "c:" << mat.size().width << endl;
    }
    

    
    for (int i = 0; i < width; ++i)
    {
        double minVal, maxVal;
        //if(DEBUG) cout << mat.row(i) << endl;
        minMaxLoc(mat.row(i), &minVal, &maxVal);
        for (int j = 0; j < 20; ++j)
        {
            _result[j][i] = maxVal;
        }
    }
    if (DEBUG) cout << "for loop done" << endl;

    Mat resultMat = Mat(20, width, CV_8UC1, &_result);

    if(DEBUG) cout << "result::r=" << resultMat.size().height\
        << "result::c" << resultMat.size().width << endl;

    applyColorMap(resultMat, resultMat, COLORMAP_JET);
    medianBlur ( resultMat, resultMat, 5 );
    //imshow("AFter medianBlur", displayMat);
    namedWindow("Distance", WINDOW_AUTOSIZE);
    imshow("Distance", resultMat);
}




