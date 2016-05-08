/**
    main.cpp
    Purpose: Checks retina image for the presence of symptoms (drusen, exudates, hemorrhages) of ARMD and DR

    @author Ján Macek
    @version 1.0 16/3/16
*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <cv.hpp>
#include <sys/stat.h>
#include<iostream>
#include<stdio.h>
#include<ctype.h>
#include<stdlib.h>
#include <getopt.h>

#include "retinaFindings.h"

int showImageCount = 0;
void showImage(std::string title, cv::Mat image) {
    uchar row = (showImageCount) / 6;
    uchar col = (showImageCount) % 6;
    showImageCount++;
    cv::namedWindow(title, cv::WINDOW_NORMAL );
    //resizeWindow(title, 300,240);
    cv::moveWindow(title, 50 + 310 * col, 30 + 265 * row);
    imshow(title, image);
    cv::waitKey(10);
}

void showImage(cv::Mat image, std::string title) {
    showImage(title, image);
}


/* @brief       Prints program help to standart output
 */
void help() {
    std::cout << " Usage: ./retinaDiseaseClasifier [ OPTIONS ... ] IMAGE" << std::endl << std::endl;
    std::cout << " -h       \tPrint this help." << std::endl;
    std::cout << " -o PATH  \tSave image with highlighted findings in it to output file." << std::endl;
    std::cout << " -m PATH  \tSpecifies path to folder, where statistical model for bayes classifier are stored." << std::endl;
    std::cout << " -d PATH  \tSpecifies path to folder, where database files with classified findings and their features are stored." << std::endl;
    std::cout << " -c [0-2] \tUser clasification of findings in retina as true positive or false positive. Number means type of findings: 0 = druse, 1 = exudate, 2 = hemorrhage." << std::endl;
    std::cout << "\t\tWith this parameter is required \"-d\" parameter." << std::endl;
    std::cout << " -t [0-2] \tRetrain classifier. Number means type of findings: 0 = druse, 1 = exudate, 2 = hemorrhage." << std::endl;
    std::cout << "\t\tWith this parameter are required \"-d\" and \"-m\" parameters." << std::endl << std::endl;
}

int main( int argc, char** argv ) {

    // check if last argument point to file, image input
    std::string imagePath;
    if(argc > 1) {
        if(argc == 2 && strcmp(argv[1], "-h") == 0) {
            help();
            return 0;
        } else {
            imagePath = argv[argc-1];
            if(!fileExist(imagePath)) {
                std::cerr << "Image file \"" << imagePath << "\" does not exist." << std::endl << std::endl;
                help();
                return 1;
            }
        }
    } else {
        std::cerr << "Image file is mandatory parameter." << std::endl;
        return 1;
    }

    int opt;
    std::string modelPath, outputPath, dataPath;
    int classification = -1, train = -1;
    while ((opt = getopt(argc-1, argv, "ctd:o:m:")) != EOF) {
        switch (opt) {
            case 'd':
                dataPath = optarg;
                if(!fileExist(dataPath)) {
                    std::cerr << "Data folder \"" << modelPath << "\" does not exist." << std::endl;
                    return 1;
                }
                break;
            case 't':
                train = atoi(optarg);
                if(train < 0 && train > 2) {
                    std::cerr << "Wrong train argument. Value is out of range [0-2]." << std::endl;
                    return 1;
                }
                break;
            case 'c':
                classification = atoi(optarg);
                if(classification < 0 && classification > 2) {
                    std::cerr << "Wrong classification argument. Value is out of range [0-2]." << std::endl;
                    return 1;
                }
                break;
            case 'm':
                modelPath = optarg;
                if(!fileExist(modelPath)) {
                    std::cerr << "Models folder \"" << modelPath << "\" does not exist." << std::endl;
                    return 1;
                }
                break;
            case 'o':
                outputPath = optarg;
                if(!fileExist(outputPath)) {
                    std::cerr << "Output image folder \"" << outputPath << "\" does not exist." << std::endl;
                    return 1;
                }
                break;
            default:
                std::cout << std::endl;
                help();
                return 1;
        }
    }

    cv::Mat image;
    image = cv::imread(imagePath, 1);
    if(!image.data ) {
        std::cerr << "Missing retina image ..." << std::endl;
        return -1;
    }

    // make black borders of image so background mask can be extracted even if image contain cutout of retina
    copyMakeBorder(image, image, 50, 50, 50, 50, cv::BORDER_CONSTANT, 0);

    if(classification != -1) {
        if (dataPath.empty()) {
            std::cerr << "Missing output for training ..." << std::endl;
            return -1;
        } else {

            // create new lines in database file
            appendDatabase(image, classification, dataPath);
        }
    } else if(train != -1) {
        if (dataPath.empty()) {
            std::cerr << "Missing path to training data folder for training ..." << std::endl;
            return -1;
        } else if (modelPath.empty()) {
            std::cerr << "Missing path to statistical model folder for training ..." << std::endl;
            return -1;
        } else {
            // update model of classifier
            trainClassifier(dataPath, modelPath, train);
        }
    } else {

        // find features in retina and draw them into image
        std::vector<std::vector<cv::Point>> contours;
        cv::Ptr<cv::ml::NormalBayesClassifier> bayes;
        cv::Mat img = image.clone();
        contours = getHemorrhagesContours(image);
        cv::Mat hemorrhages(image.size(), CV_8UC1, cv::Scalar(0));
        // if statistical model for bayes exists, use it and predict if hemorrhage is true positive finding
        if(!modelPath.empty() && fileExist(modelPath + "/" + HEMORRHAGE_MODEL_NAME)) {
            bayes = cv::ml::StatModel::load<cv::ml::NormalBayesClassifier>(modelPath);
            for(int i = 0; i < contours.size(); i++) {
                std::vector<double> features = getHemorrhageFeature(contours, &image, i);
                if(bayes->predict(features) == 1) {
                    drawContours( img, contours, i, cv::Scalar(66, 41, 255), 1, 8);
                    drawContours( hemorrhages, contours, i, 255, CV_FILLED, 8);
                };
            }
        } else {
            for(int i = 0; i < contours.size(); i++) {
                drawContours( img, contours, i, cv::Scalar(66, 41, 255), 1, 8);
                drawContours( hemorrhages, contours, i, 255, CV_FILLED, 8);
            }
        }


        // if hemorrhages are present, then search for exudates
        if(cv::countNonZero(hemorrhages) > 0.01 * cv::countNonZero(getBackgroundMask(image, 0, 5))){
            contours = getExudatesContours(image);
            // if statistical model for bayes exists, use it and predict if exudate is true positive finding
            if(!modelPath.empty() && fileExist(modelPath + "/" + EXUDATE_MODEL_NAME)) {
                bayes = cv::ml::StatModel::load<cv::ml::NormalBayesClassifier>(modelPath);
                for(int i = 0; i < contours.size(); i++) {
                    std::vector<double> features = getBrightObjectFeature(contours, &image, i);
                    if(bayes->predict(features) == 1) {
                        drawContours( img, contours, i, cv::Scalar(255, 0, 0), 1, 8);
                    };
                }
            } else {
                for(int i = 0; i < contours.size(); i++) {
                    drawContours( img, contours, i, cv::Scalar(255, 0, 0), 1, 8);
                }
            }
            if(!contours.empty()){
                std::cout << "There are bright findings in retina. Because hemorrhages are present, they are classified as exudates." << std::endl;
            }
        // if hemorrhages are not present, then search for drusen
        } else {
            contours = getDrusenContours(image);
            // if statistical model for bayes exists, use it and predict if druse is true positive finding
            if(!modelPath.empty() && fileExist(modelPath + "/" + DRUSE_MODEL_NAME)) {
                bayes = cv::ml::StatModel::load<cv::ml::NormalBayesClassifier>(modelPath);
                for(int i = 0; i < contours.size(); i++) {
                    std::vector<double> features = getBrightObjectFeature(contours, &image, i);
                    if(bayes->predict(features) == 1) {
                        drawContours( img, contours, i, cv::Scalar(255, 0, 0), 1, 8);
                    };
                }
            } else {
                for(int i = 0; i < contours.size(); i++) {
                    drawContours( img, contours, i, cv::Scalar(255, 0, 0), 1, 8);
                }
            }
            std::cout << "There are bright findings in retina. Because hemorrhages are not present of there is only small amount of them, bright findings are classified as drusen." << std::endl;
        }

        // save image with drawn features to image
        if(!outputPath.empty() && !fileExist(outputPath)){
            // get name of image from image path, prepend "findings-"
            std::string fileName = (imagePath.find("/") > 0) ? imagePath.substr(imagePath.find_last_of("/") + 1) : imagePath;
            fileName = "-findings" + fileName;
            if(outputPath.back() == '/') outputPath.pop_back();
            cv::imwrite((outputPath + "/" + fileName), image);
        // show image in window
        } else {
            std::string windowTitle = "result";
            cv::namedWindow(windowTitle, cv::WINDOW_NORMAL );
            cv::resizeWindow(windowTitle, 700,700);
            cv::moveWindow(windowTitle, 50,50);
            cv::imshow(windowTitle, img);
            cv::waitKey(0);
        }
    }

    return 0;
}