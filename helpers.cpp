/**
    helpers.cpp
    Purpose: Provides helper function for programing with openCV library

    @author Ján Macek
    @version 1.0 16/3/16
*/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "helpers.h"

using namespace cv;
using namespace std;

int showImageCount = 0;

void showImage(String title, Mat image){
    uchar row = (showImageCount) / 6;
    uchar col = (showImageCount) % 6;
    showImageCount++;
    namedWindow(title, WINDOW_NORMAL );
    //resizeWindow(title, 300,240);
    moveWindow(title, 50 + 310 * col, 30 + 265 * row);
    imshow(title, image);
    cv::waitKey(10);
}

void showImage(Mat image, String title){
    showImage(title, image);
}



