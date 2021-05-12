#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <ctime>

using namespace cv;

Mat rgbImageL, rgbImageR;

long long getTimeStamp()
{
    const std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    const std::chrono::microseconds epoch = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return epoch.count();
}

std::string folder_name = "";

int main()
{
    VideoCapture cam0("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)10/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    VideoCapture cam1("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)10/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
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

    long long frame_number = 0;
    long long start_time = getTimeStamp();
    long long total_time = 0;
    

    while(1)
    {
        cam0 >> rgbImageL;
        cam1 >> rgbImageR;

        imshow("Image Left", rgbImageL);
        imshow("Image Right", rgbImageR);

        frame_number++;

        char k  = waitKey(1);
        if (k == 'q' || k == 'Q') break;

    }

    total_time = getTimeStamp();
    float fps = frame_number / ((total_time - start_time) / 1000000);
    fprintf(stdout, "Frames number: %lld\nTotal time: %lld\nAverage FPS: %f\n", frame_number, total_time - start_time, fps);

    cam0.release();
    cam1.release();
    destroyAllWindows();
    return 0;

}