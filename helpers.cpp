/**
    helpers.cpp
    Purpose: Provides helper function for programing with openCV library

    @author Ján Macek
    @version 1.0 16/3/16
*/

#include <opencv2/core/cvdef.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "helpers.h"

int showImageCount = 0;

void showImage(std::string title, cv::Mat image){
    uchar row = (showImageCount) / 6;
    uchar col = (showImageCount) % 6;
    showImageCount++;
    cv::namedWindow(title, cv::WINDOW_NORMAL );
    //resizeWindow(title, 300,240);
    cv::moveWindow(title, 50 + 310 * col, 30 + 265 * row);
    imshow(title, image);
    cv::waitKey(10);
}

void showImage(cv::Mat image, std::string title){
    showImage(title, image);
}



