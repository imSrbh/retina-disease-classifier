/**
    main.cpp
    Purpose: Checks fundus image for the presence of symptoms (drusen, exudates, hemorrhages, neovascularization) of ARMD and DR

    @author Ján Macek
    @version 1.0 16/3/16
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/ml/ml.hpp>

#include "helpers.h"

const int MASK_CONTOURS_LINE_THICKNESS = 7;
const int DEFAULT_OPTIC_DISC_SIZE = 60;
const int S1 = 5;
const int S2 = 10;
const int KERNEL_BRANCH_SIZE = 5;

const int MAX_CLUSTERING_ITERATIONS = 200;
const int MAX_CLUSTERING_DISTANCE = 40;
const double MAX_MACULA_ECCENTRICITY = 0.5;

const int BG_MASK_BY_CONTOURS = 0;
const int BG_MASK_BY_LOCAL_VARIANCE = 1;

const int LEARN_DRUSE = 0;
const int LEARN_EXUDATE = 1;
const int LEARN_HEMORRHAGE = 2;

/*  @brief  Structure that represents settings of gabor filter kernel.
 */
struct GaborFilterParams {
    double sigma;
    double theta;
    double lambda;
    double gamma;
    double psi;
    int kernelSize;
};

const std::vector<GaborFilterParams> GABOR_FILTER_BANK = {
        {5,   0, 8,  1,     0, 70},
        {2,   0, 15, 0.7,   0, 70},
        {10,  0, 6,  0,     0, 70},
        {20,  0, 7,  0.15,  0, 30},
        {1.6, 0, 16, 0.16,  0, 30} };

/*  @brief  Structure that represents features of contour/element.
 */
struct ContourFeatures {
    double area;
    double compactness;
    double avgBoundaryIntensity;
    double minBoundaryIntensity;
    double maxBoundaryIntensity;
    double meanHue;
    double meanSaturation;
    double meanIntensity;
    double meanGradientMagnitude;
    double energy;
    double entropy;
    double ratio;
};

/*  @brief      Converts druse struct to matrix of druse values.
 *  @param      structure of druse
 *  @return     one line matrix of eleven double values
 */
cv::Mat druseToMat(ContourFeatures druse) {
    double data[] = {
            druse.area,
            druse.compactness,
            druse.avgBoundaryIntensity,
            druse.minBoundaryIntensity,
            druse.maxBoundaryIntensity,
            druse.meanHue,
            druse.meanSaturation,
            druse.meanIntensity,
            druse.meanGradientMagnitude,
            druse.energy,
            druse.entropy,
            druse.ratio
    };
    cv::Mat result(1, 12, CV_64FC1, (double*)data);
    return result;
}

/*  @brief     Creates structuring element in shape of hexagon where object image points are set to 255 and others are set to 0.
 *  @param     size is the length of one hexagon edge
 *  @return    matrix of hexagonal element
 */
cv::Mat getHexagonalStructuringElement(int size) {
    cv::Mat elem; elem = cv::Mat::zeros(size * 2 - 1, size * 3 - 2, CV_8UC1);
    int oneCount = size, zeroCount = size - 1;
    for(int row = 0; row < elem.rows; row++) {
        uchar *imagePtr = elem.ptr(row);
        for (int j = 0; j < elem.cols; ++j) {
            if(j >= zeroCount && j < zeroCount + oneCount){
                *imagePtr = 255;
            } else {
                *imagePtr = 0;
            }
            *imagePtr++;
        }
        if(row < size - 1){
            zeroCount--;
            oneCount+=2;
        } else {
            zeroCount++;
            oneCount-=2;
        }
    }
    return elem;
}

/*  @brief      Executes reconstruction of image by morfological operation while image before and after operation are not same.
 *  @param      mask of skipped image points, its non-zero elements indicate which matrix elements need to be processed
 *  @param      toReconstruct is matrix which is going to be processed
 *  @param      structureElem is matrix element by which morfological operation is computed
 *  @param      type of morfological operation: 0 = dilatation, other = erosion)
 *  @return     reconstructed image
 */
cv::Mat morfologicalReconstruction(cv::Mat mask, cv::Mat toReconstruct, cv::Mat structureElem, int type) {
    cv::Mat m0, m1 = toReconstruct.clone(), ne;
    do {
        m0 = m1.clone();
        if(type == 0){
            dilate(m0, m1, structureElem);
            cv::min(m1, mask, m1);
        } else {
            erode(m0, m1, structureElem);
            cv::max(m1, mask, m1);
        }
        cv::min(m1, mask, m1);
        cv::compare(m1, m0, ne, cv::CMP_NE);
    } while(cv::countNonZero(ne) != 0);
    return m1;
}

/*  @brief      Computes ratio of loaded image and images used to test functions.
 *  @param      original are dimensions of loaded image
 *  @param      pattern are default dimensions of test image
 *  @return     double value of ratio
 */
double computeRatio(cv::Size original, cv::Size pattern = {480, 640}) {
    return ((double)(original.height / pattern.height) + (double)(original.width / pattern.width)) / 2.0;
}

/*  @brief      Compute correction of shades at image.
 *  @param      image to be processed
 *  @param      ratio is optional parameter, if not specified, new is computed by image parameter
 *  @return     matrix of shade corrected image
 */
cv::Mat shadeCorrection(cv::Mat image, double ratio = std::numeric_limits<double>::infinity()) {
    cv::Mat structure_elem;

    if(ratio == std::numeric_limits<double>::infinity()) {
        ratio = computeRatio({image.cols, image.rows});
    }

    if(image.channels() > 1) {
        cv::cvtColor(image, image, CV_BGR2GRAY);
    }

    int maxStructureSize = 15 * ratio, structureSize = 1;

    while(structureSize < maxStructureSize){
        structure_elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(structureSize, structureSize));
        //morphological opening
        morphologyEx(image, image, cv::MORPH_OPEN, structure_elem);
        //morphological closing
        morphologyEx(image, image, cv::MORPH_CLOSE, structure_elem);
        //double structure size
        structureSize *= 2;
    }

    return image;
}

/*  @brief      Computes standard local variation of matrix of input image.
 *  @param      image to be processed
 *  @param      mask of skipped image points, its non-zero elements indicate which matrix elements need to be processed
 *  @param      kernelSize define size of structuring element used for computing
 *  @param      max contains coordinates of point with the highest value of local variation
 *  @return     matrix of image local variation
 */
cv::Mat localVariation(cv::Mat image, cv::Mat mask, int kernelSize, cv::Point *max = nullptr) {

    // convert input image to grayscale
    if(mask.channels() == 3) {
        cvtColor(mask, mask, CV_BGR2GRAY );
    }

    // make borders to input image and image mask, to handle out of image dimension values when processing
    cv::Mat imageBorder;
    copyMakeBorder(image, imageBorder, KERNEL_BRANCH_SIZE, KERNEL_BRANCH_SIZE, KERNEL_BRANCH_SIZE, KERNEL_BRANCH_SIZE, cv::BORDER_REFLECT);
    cv::Mat maskBorder;
    copyMakeBorder(mask, maskBorder, KERNEL_BRANCH_SIZE, KERNEL_BRANCH_SIZE, KERNEL_BRANCH_SIZE, KERNEL_BRANCH_SIZE, cv::BORDER_CONSTANT, 0);

    // prepare matrix for result of local variation
    cv::Mat localVariationImage;
    localVariationImage = cv::Mat::ones(image.rows, image.cols, CV_8UC1);

    double maxValue = 0;

    // for each point of each row is local variation computed
    for(int row = 0; row < (localVariationImage).rows; row++) {

        // get lines of matrices
        uchar *imagePtr = (localVariationImage).ptr(row);
        uchar *maskPtr = (mask).ptr(row);

        for (int col = 0; col < (localVariationImage).cols; col++) {

            // if this point has non-zero value in mask
            if(*maskPtr != 0){

                // get kernel with actual point in the middle and predefined edge size
                cv::Mat imageKernel = imageBorder(cv::Rect(col, row, kernelSize, kernelSize));
                cv::Mat maskKernel = maskBorder(cv::Rect(col, row, kernelSize, kernelSize));

                // get mean of kernel
                cv::Scalar kernelMeanScalar, kernelStdDevScalar;
                meanStdDev(imageKernel, kernelMeanScalar, kernelStdDevScalar);

                // convert kernel to one channel image
                cv::Mat imageKernelOneChannel;
                imageKernel.convertTo(imageKernelOneChannel, CV_8UC1);

                // subtract mean value of all elements in kernel matrix
                subtract(imageKernelOneChannel, cv::Scalar(kernelMeanScalar[0]), imageKernelOneChannel);

                // exponentiate all values in kernel matrix
                pow(imageKernelOneChannel, 2, imageKernelOneChannel);

                // remove/reset all masked values in kernel matrix
                cv::Mat maskedImageKernel;
                imageKernelOneChannel.copyTo(maskedImageKernel, maskKernel);

                // compute local variation of pixel
                double finalValue = (sum(maskedImageKernel)[0])/(pow(kernelSize, 2)-1.0);
                *imagePtr = finalValue;

                // recompute max value position
                if(max != nullptr && maxValue < finalValue) {
                    *max = { col, row };
                    maxValue = finalValue;
                }
            }
            *maskPtr++;
            *imagePtr++;
        }
    }
    return localVariationImage;
}

/*  @brief      Finds retina image background mask by local variance of image or by contours of image
 *  @param      image to be processed
 *  @param      method - allowed values are BG_MASK_BY_CONTOURS = 0, BG_MASK_BY_LOCAL_VARIANCE = 1
 *  @param      reduction is optional parameter, define reduction of dimensions of foreground image
 *  @param      ratio is optional parameter, if not specified, new is computed by image parameter
 *  @return     matrix with same size as original image, zero element are background elements
 */
cv::Mat getBackgroundMask(cv::Mat image, int method, int reduction, double ratio = std::numeric_limits<double>::infinity()) {

    if(method != BG_MASK_BY_CONTOURS && method != BG_MASK_BY_LOCAL_VARIANCE) {
        throw std::invalid_argument("Invalid 'method' property value. Enabled only BG_MASK_BY_CONTOURS = 0, BG_MASK_BY_LOCAL_VARIANCE = 1.");
    }

    if(ratio == std::numeric_limits<double>::infinity()) {
        ratio = computeRatio({image.cols, image.rows});
    }

    if(image.channels() == 3){
        cvtColor(image, image, cv::COLOR_BGR2GRAY );
    }

    cv::Mat computedImage;

    if( method == BG_MASK_BY_LOCAL_VARIANCE ) {
        // compute variance of image
        image.convertTo(image, CV_32F);
        cv::Mat mu;
        blur(image, mu, cv::Size(10, 10));

        cv::Mat mu2;
        blur(image.mul(image), mu2, cv::Size(10, 10));

        cv::subtract(mu2, mu.mul(mu), image);
        cv::sqrt(image, image);

        cv::Mat thresh;
        threshold(image, thresh, 10, 255, cv::THRESH_BINARY);
        thresh.convertTo(computedImage, CV_8UC1);

    } else if( method == BG_MASK_BY_CONTOURS){
        // apply canny edge detector
        cv::Mat canny;
        Canny(image, canny, 20, 60);

        // find the contours
        std::vector<std::vector<cv::Point>> contours;
        findContours(canny, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        // draw contours to image, thickness depends on image size
        cv::Mat result; result = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
        for( size_t i = 0; i< contours.size(); i++ ) {
            drawContours( result, contours, (int)i, cv::Scalar(255,255,255), 3 * ratio, 8);
        }
        cvtColor(result, computedImage, cv::COLOR_BGR2GRAY );
    }

    // find the largest contour of image, largest is object of retina
    std::vector<std::vector<cv::Point>> contours;
    findContours(computedImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    int largestArea = 0;
    int largestContourIndex = 0;
    for( int i = 0; i< contours.size(); i++ ) {
        //Find the area of contour
        double areaIndex = contourArea(contours[i], false);
        if(areaIndex > largestArea) {
            largestArea = areaIndex;
            largestContourIndex =i;
        }
    }

    // fill contour and return it as black / white image
    cv::Mat mask(image.rows, image.cols, CV_8UC1, cv::Scalar(0, 0, 0));
    drawContours(mask, contours, largestContourIndex, cv::Scalar(255, 255, 255), (MASK_CONTOURS_LINE_THICKNESS + reduction) * ratio, 8);
    floodFill(mask, cv::Point(5, 5), CV_RGB(255, 255, 255));
    bitwise_not(mask, mask);

    return mask;
}

/*  @brief      Finds optic disc center by standard local variation.
 *  @param      image to be processed
 *  @param      subImageSize define size of sub-image, which is used for improving center position of optic disc by distance tranformation
 *  @param      ratio is optional parameter, if not specified, new is computed by image parameter
 *  @return     point which points to optic disc center
 */
cv::Point getOpticDiscCenter(cv::Mat image, int subImageSize, double ratio = std::numeric_limits<double>::infinity()) {

    if(ratio == std::numeric_limits<double>::infinity()){
        ratio = computeRatio({image.cols, image.rows});
    }

    // kernel width is computed as center point plus branch size twice, each side = left and right / top and bottom
    int kernelSize = KERNEL_BRANCH_SIZE * 2 + 1;

    // for localization of optic disc is the best option luminance channel
    cv::Mat yuvImage, yuvChannels[3];
    cvtColor(image, yuvImage, CV_BGR2YUV);
    split(yuvImage, yuvChannels);
    cv::Mat luminanceChannel = yuvChannels[0];

    // remove small background variations by applying shade-correction operator
    cv::Mat shadeCorrectedImage = shadeCorrection(luminanceChannel, ratio);

    // get background mask to exclude retina border pixels
    cv::Mat mask = getBackgroundMask(shadeCorrectedImage, 0, 5, ratio);

    // get pixel with maximum value of local variation
    cv::Point maxLocalVariationPosition;
    cv::cvtColor(mask, mask, CV_GRAY2BGR);
    localVariation(shadeCorrectedImage, mask, kernelSize, &maxLocalVariationPosition);

    // cut out sub-image with optic disc centroid by local variation in its center
    cv::Mat opticDiscSubImage = image(cv::Rect(maxLocalVariationPosition.x - (0.5*subImageSize), maxLocalVariationPosition.y - (0.5*subImageSize), subImageSize, subImageSize));
    cv::cvtColor(opticDiscSubImage, opticDiscSubImage, CV_BGR2GRAY);

    // compute 80% threshold of gray color optic disc sub-image
    double min, max;
    minMaxLoc(opticDiscSubImage, &min, &max);
    threshold(opticDiscSubImage, opticDiscSubImage, max * 0.8, 255, cv::THRESH_BINARY );

    // find biggest object of threshold sub-image
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours( opticDiscSubImage, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
    int largestArea =0;
    int largestContourIndex =0;
    for( int i = 0; i< contours.size(); i++ ) {
        // find the area of contour
        double areaIndex = contourArea(contours[i], false);
        if(areaIndex > largestArea) {
            largestArea = areaIndex;
            largestContourIndex =i;
        }
    }

    // draw the biggest object using previously computed contour index
    cv::Mat biggestTresholdParticle(opticDiscSubImage.rows, opticDiscSubImage.cols, CV_8UC1, cv::Scalar::all(0));
    drawContours(biggestTresholdParticle, contours, largestContourIndex, cv::Scalar(255,255,255), CV_FILLED, 8, hierarchy );

    // find real optic disc center as max of largest object in distance transformation of threshold image
    cv::Mat distanceImage;
    distanceTransform(biggestTresholdParticle, distanceImage, CV_DIST_L2, 3);
    //normalize(distanceImage, distanceImage, 0, 1., cv::NORM_MINMAX);
    cv::Point min_loc, max_loc;
    minMaxLoc(distanceImage, &min, &max, &min_loc, &max_loc);

    // normalize max point sub-image coordinates to coordinates of entire image
    return  {max_loc.x + maxLocalVariationPosition.x, max_loc.y + maxLocalVariationPosition.y};
}

/*  @brief      Finds optic disc contours by watershed transformation.
 *  @param      image to be processed
 *  @param      ratio is optional parameter, if not specified, new is computed by image parameter
 *  @return     vector of contours ( vector of points ) of optic disc
 */
std::vector<std::vector<cv::Point>> getOpticDiscContours(cv::Mat image, double ratio = std::numeric_limits<double>::infinity()) {

    if(ratio == std::numeric_limits<double>::infinity()){
        ratio = computeRatio({image.cols, image.rows});
    }

    // sub-image where optic disc can be find is triple size of optic disc size based on literature, because center find by us can be on the border of optic disc
    int subImageSize = (DEFAULT_OPTIC_DISC_SIZE*ratio) * 3;
    cv::Point opticDiscCenter = getOpticDiscCenter(image, subImageSize, ratio);
    cv::Mat opticDiscSubImage = image(cv::Rect(opticDiscCenter.x - subImageSize, opticDiscCenter.y - subImageSize, subImageSize, subImageSize));

    // red channel is the best of channels to find contours of optic disc
    std::vector<cv::Mat> bgrChannel;
    split(opticDiscSubImage, bgrChannel);
    cv::Mat redChannel = (bgrChannel[2]);
    normalize(redChannel, redChannel, 0, 255, cv::NORM_MINMAX);



    // fill the vessels, applying a simple closing
    cv::Mat structureElem = getHexagonalStructuringElement(S1 * ratio);
    cv::Mat closedRedChannel;
    morphologyEx(redChannel, closedRedChannel, cv::MORPH_CLOSE, structureElem);

    // in order to remove large peaks, we open  with a large structuring element
    structureElem = getHexagonalStructuringElement(S2 * ratio);
    cv::Mat openRedChannel;
    morphologyEx(closedRedChannel, openRedChannel, cv::MORPH_OPEN, structureElem);

    // compute morfological reconstruction of closed red channel by opened red channel;
    cv::Mat morfRedChannel = morfologicalReconstruction(closedRedChannel, openRedChannel, structureElem, 0);

    // create markers image to prevent over-segmentation in watershed transformation
    cv::Mat markers(morfRedChannel.size(),CV_8U,cv::Scalar(-1));
    circle(markers, cv::Point(morfRedChannel.rows/2, morfRedChannel.cols/2), redChannel.rows/2,  cv::Scalar::all(1), 1);
    circle(markers, cv::Point(morfRedChannel.rows/2, morfRedChannel.cols/2), 1,  cv::Scalar::all(2), 1);
    markers.convertTo(markers, CV_32S);

    // to find contours, process watershed transformation and threshold of its result
    cvtColor(morfRedChannel, morfRedChannel, CV_GRAY2RGB);
    cv::watershed(morfRedChannel, markers);
    cv::Mat mask;
    convertScaleAbs(markers, mask, 1, 0);
    threshold(mask, mask, 1, 255, cv::THRESH_BINARY);
    double top = opticDiscCenter.x - subImageSize;
    double left = opticDiscCenter.y - subImageSize;
    copyMakeBorder(mask, mask,  left, image.cols - left, top, image.rows - top,0);

    // find the contours
    std::vector<std::vector<cv::Point>> contours;
    findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    return contours;
}

/* @brief       Creates mask of optic disc.
 * @param       image to be processed
 * @param       reduction is optional parameter, define reduction of dimensions of foreground image
 * @param       ratio is optional parameter, if not specified, new is computed by image parameter
 * @return      matrix with same size as original image, zero element are optic disc elements
 */
cv::Mat getOpticDiscMask(cv::Mat image, int reduction, double ratio = std::numeric_limits<double>::infinity()) {

    if(ratio == std::numeric_limits<double>::infinity()){
        ratio = computeRatio({image.cols, image.rows});
    }

    std::vector<std::vector<cv::Point>> contours = getOpticDiscContours(image, ratio);

    cv::Mat mask(image.rows, image.cols, CV_8UC1, cv::Scalar(255, 255, 255));
    for(size_t i = 0; i< contours.size(); i++ ) {
        drawContours(mask, contours, (int)i, cv::Scalar(0, 0, 0), (2 + reduction) * ratio, 1);
    }

    bitwise_not(mask, mask);
    floodFill(mask, cv::Point(5, 5), CV_RGB(255, 255, 255));

    return mask;
}

/*  @brief      Computes histogram equalization of intensity channel.
 *  @param      image to be processed
 *  @return     image with equalized intensity
 */
cv::Mat equalizeIntensity(cv::Mat image) {

    cv::Mat ycrcbImage;
    cvtColor(image, ycrcbImage, CV_BGR2YCrCb);

    std::vector<cv::Mat> ycrcbChannels;
    split(ycrcbImage, ycrcbChannels);

    equalizeHist(ycrcbChannels[0], ycrcbChannels[0]);

    cv::Mat result;
    merge(ycrcbChannels, ycrcbImage);
    cvtColor(ycrcbImage, result, CV_YCrCb2BGR);

    return result;
}

/*  @brief      Computes adaptive histogram equalization of color image.
 *  @param      image to be processed
 *  @param      clipLimit is threshold for contrast limiting
 *  @param      tileGridSize is size of grid for histogram equalization
 */
cv::Mat clahe(cv::Mat image, double clipLimit, cv::Size tileGridSize) {

    cv::Mat lab_image;
    cv::cvtColor(image, lab_image, CV_BGR2Lab);

    // Extract the L channel
    std::vector<cv::Mat> lab_planes(3);
    // now we have the L image in lab_planes[0]
    cv::split(lab_image, lab_planes);

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, tileGridSize);
    cv::Mat dst;
    clahe->apply(lab_planes[0], dst);

    // Merge the the color planes back into an Lab image
    dst.copyTo(lab_planes[0]);
    cv::merge(lab_planes, lab_image);

    // convert back to BGR color color
    cv::Mat image_clahe;
    cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);

    return image_clahe;
}

/* @brief       Generates Gabor filter imaginary coefficients.
 * @param       ksize is size of the filter returned
 * @param       sigma is tandard deviation of the gaussian envelope
 * @param       theta is orientation of the normal to the parallel stripes of a Gabor function
 * @param       lambda is wavelength of the sinusoidal factor
 * @param       gamma is spatial aspect ratio
 * @param       psi is phase offset
 * @param       ktype is type of filter coefficients. It can be CV_32F or CV_64F
 * @return      generated matrix of gabor kernel
 */
cv::Mat getGaborKernelImaginary(cv::Size ksize, double sigma, double theta, double lambda, double gamma, double psi, int ktype = CV_64F) {
    double sigma_x = sigma;
    double sigma_y = sigma/gamma;
    int nstds = 3;
    int xmin, xmax, ymin, ymax;
    double c = cos(theta), s = sin(theta);

    if( ksize.width > 0 ) {
        xmax = ksize.width / 2;
    } else {
        xmax = cvRound(std::max(fabs(nstds * sigma_x * c), fabs(nstds * sigma_y * s)));
    }

    if( ksize.height > 0 ) {
        ymax = ksize.height/2;
    } else {
        ymax = cvRound(std::max(fabs(nstds * sigma_x * s), fabs(nstds * sigma_y * c)));
    }

    xmin = -xmax;
    ymin = -ymax;

    CV_Assert( ktype == CV_32F || ktype == CV_64F );

    cv::Mat kernel(ymax - ymin + 1, xmax - xmin + 1, ktype);
    double scale = 1;
    double ex = -0.5/(sigma_x*sigma_x);
    double ey = -0.5/(sigma_y*sigma_y);
    double cscale = CV_PI*2/lambda;

    for( int y = ymin; y <= ymax; y++ ) {
        for (int x = xmin; x <= xmax; x++) {
            double xr = x * c + y * s;
            double yr = -x * s + y * c;

            double v = scale * std::exp(ex * xr * xr + ey * yr * yr) * (cos(cscale * xr + psi));
            if (ktype == CV_32F)
                kernel.at<float>(ymax - y, xmax - x) = (float) v;
            else
                kernel.at<double>(ymax - y, xmax - x) = v;
        }
    }

    return kernel;
}

/* @brief       Finds exudates contours in retina image.
 * @param       image to be processed
 * @param       opticDiscMask is mask of pixels where is optic disc, if it is empty ( all elemets of matrix are zeros), new one is computed in image
 * @param       ratio is optional parameter, if not specified, new is computed by image parameter
 * @return      vector of contours ( vector of points ) of exudates
 */
std::vector<std::vector<cv::Point>> getExudatesContours(cv::Mat image, cv::Mat opticDiscMask = cv::Mat(), double ratio = std::numeric_limits<double>::infinity()) {

    if(ratio == std::numeric_limits<double>::infinity()){
        ratio = computeRatio({image.cols, image.rows});
    }

    if(opticDiscMask.empty()){
        cv::Mat structureElem = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(10 * ratio, 10 * ratio));
        opticDiscMask = getOpticDiscMask(image, 0, ratio);
        erode(opticDiscMask, opticDiscMask, structureElem);
    }

    // exudates appear most contrasted in green channel
    std::vector<cv::Mat> bgrChannel;
    split(image, bgrChannel);
    cv::Mat greenChannel = bgrChannel[1];

    // eliminate the vessels by a morfological closing, S1 is maximal width of blood vessel
    cv::Mat structureElem = getStructuringElement(CV_SHAPE_RECT, cv::Size(S1 * ratio, S1 * ratio)), morfGreenChannel;
    morphologyEx(greenChannel, morfGreenChannel, cv::MORPH_CLOSE, structureElem);

    // calculate the local variation for each pixel
    int kernelSize = KERNEL_BRANCH_SIZE * 2 + 1;
    cv::Mat backgroundMask = getBackgroundMask(morfGreenChannel, 0, ratio);
    cv::Mat localVariationImage = localVariation(morfGreenChannel, backgroundMask, kernelSize);
    normalize(localVariationImage, localVariationImage, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // thresholding local varion image to extract candidate region, A1 is chosen in a very tolerant manner
    int A1 = 10;
    threshold(localVariationImage, localVariationImage, A1, 255, cv::THRESH_BINARY );

    // fill inner areas of candidate regions
    cv::Mat filled; filled = localVariationImage.clone();
    floodFill(filled, cv::Point(0, 0), 255);
    bitwise_not(filled, filled);
    bitwise_or(filled, localVariationImage, filled);

    // merge mask of optic disc and background mask
    cv::Mat mask; mask = cv::Mat::ones(image.rows, image.cols, CV_8UC1);
    bitwise_and(opticDiscMask, backgroundMask, mask);

    // remove the candidate region that results from the optic disc or are in background
    cv::Mat candidateRegions; candidateRegions = cv::Mat::ones(image.rows, image.cols, CV_8UC1);
    filled.copyTo(candidateRegions, mask);

    // set all the candidate regions to 0 in the original image
    cv::Mat e6; e6 = cv::Mat::ones(image.rows, image.cols, CV_8UC1);
    bitwise_not(candidateRegions,candidateRegions);
    greenChannel.copyTo(e6, candidateRegions);

    // calculate the morphological reconstruction by dilation
    cv::Mat e7 = morfologicalReconstruction(greenChannel, e6, structureElem, 0);

    // apply a simple threshold operation to the difference between the original image and the reconstructed image
    cv::Mat efin;
    subtract(greenChannel, e7, efin);
    int A2 = 5;
    threshold( efin, efin, A2, 255, cv::THRESH_BINARY );

    // find the contours in final image
    std::vector<std::vector<cv::Point>> contours;
    findContours(efin, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    return contours;
}

/* @brief       Finds contours of drusen in image of retina.
 * @param       image to be processed
 * @param       ratio is optional parameter, if not specified, new is computed by image parameter
 * @return      vector of contours ( vector of points ) of drusen
 */
std::vector<std::vector<cv::Point>> getDrusenContours(cv::Mat image, double ratio = std::numeric_limits<double>::infinity()) {

    if (ratio == std::numeric_limits<double>::infinity()) {
        ratio = computeRatio({image.cols, image.rows});
    }

    // suppress dark regions by application of morphological closing
    cv::Mat structureElem = cv::getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(35, 35));
    cv::Mat preprocessedImage;
    morphologyEx(image, preprocessedImage, cv::MORPH_CLOSE, structureElem);

    //improve the contrast by adaptive histogram equalization
    preprocessedImage = clahe(preprocessedImage, 1, {20, 20});
    cv::cvtColor(preprocessedImage, preprocessedImage, CV_BGR2GRAY);

    // merge responses of chosen gabor filters
    cv::Mat responses;
    responses = cv::Mat::zeros(image.rows, image.cols, CV_32F);
    for (int j = 0; j < GABOR_FILTER_BANK.size(); ++j) {
        for (int i = 0; i < 360; i += 45) {

            // create gabor kernel and get response of filter
            auto gaborKernel = getGaborKernelImaginary(
                    cv::Size(GABOR_FILTER_BANK[j].kernelSize, GABOR_FILTER_BANK[j].kernelSize),
                    GABOR_FILTER_BANK[j].sigma,
                    GABOR_FILTER_BANK[j].theta + i / 180. * M_PI,
                    GABOR_FILTER_BANK[j].lambda,
                    GABOR_FILTER_BANK[j].gamma,
                    GABOR_FILTER_BANK[j].psi);

            cv::Mat filterResponse;
            cv::filter2D(preprocessedImage, filterResponse, CV_32F, gaborKernel);

            // get maximum of filter response by applying adaptive threshold
            filterResponse.convertTo(filterResponse, CV_8UC1, 1.0 / 255.0);
            adaptiveThreshold(filterResponse, filterResponse, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                              31, 0);

            // join result to other responses
            filterResponse.convertTo(filterResponse, CV_32F);
            responses += filterResponse;
        }
    }

    cv::normalize(responses, responses, 0, 255, cv::NORM_MINMAX);
    responses.convertTo(responses, CV_8U);


    // get candidate regions as dilated maximum of responses of gabor filter
    cv::Mat candidateRegions;
    threshold(responses, candidateRegions, 85, 255, cv::THRESH_BINARY);
    structureElem = cv::getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(20, 20));
    dilate(candidateRegions, candidateRegions, structureElem);

    // from candidate regions are removed regions of optic disc and regions outside of retina (in background)
    cv::Mat trueCandidateRegions(image.rows, image.cols, CV_8UC1, cv::Scalar(0, 0, 0));
    cv::Mat backgroundMask = getBackgroundMask(image, 1, 5, ratio);
    cv::Mat opticDiscMask = getOpticDiscMask(image, 5, ratio);
    cv::Mat mask;
    cv::bitwise_and(opticDiscMask, backgroundMask, mask);
    candidateRegions.copyTo(trueCandidateRegions, mask);

    // to find contours fill candidate regions in original image by black color and thet compute morfological reconstruction
    cv::Mat imageWithCandidateDeleted;
    imageWithCandidateDeleted = cv::Mat::ones(image.size(), CV_8UC1);
    cv::bitwise_not(trueCandidateRegions, trueCandidateRegions);
    image.copyTo(imageWithCandidateDeleted, trueCandidateRegions);
    cv::Mat imageGray;
    cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
    cvtColor(imageWithCandidateDeleted, imageWithCandidateDeleted, cv::COLOR_BGR2GRAY);
    cv::Mat reconstructed = morfologicalReconstruction(imageGray, imageWithCandidateDeleted, structureElem, 0);

    // get difference of original and reconstructed image, use threshold filter to get true druse regions and find contours of these regions
    cv::Mat drusen;
    subtract(imageGray, reconstructed, drusen);
    int A2 = 5;
    threshold(drusen, drusen, A2, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    findContours(drusen, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    return contours;
}

/* @brief       Computes entropy of image by histogram equalization.
 * @param       image to be processed
 * @param       histMax is maximum value of histogram
 * @param       mask of skipped image points, its non-zero elements indicate which matrix elements need to be processed
 * @return      entropy of image
 */
float computeEntropy(cv::Mat image, int histMax, cv::Mat mask) {

    // calculate relative occurrence of different symbols within given input sequence using histogram
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    cv::Mat hist;

    // compute the histograms
    calcHist(&image, 1, 0, mask, hist, 1, &histMax, &histRange, uniform, accumulate );

    int cnt = 0;
    float entropy = 0;
    // total size of all symbols in an image
    float total_size = image.size().height * image.size().width;

    for(int i=0; i < histMax; i++) {
        // the number of times a symbol has occurred
        float sym_occur = hist.at<float>(0, i);
        // log of zero goes to infinity
        if(sym_occur > 0) {
            cnt++;
            entropy += (sym_occur/total_size)*(log2(total_size/sym_occur));
        }
    }
    return entropy;
}

/*  @brief  Structure that encapsulates all attributes that are necessary in callbackFunction for learning.
 */
struct CallBackParams {
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat image;
    std::string windowName;
    std::vector<int> selectedIndexes;
};

/* @brief       Finds index of contour in which the point is.
 * @param       contours is pointer to array of elements contours
 * @param       dimensions are width and height of image from which contours are
 * @param       position is point in image, if out of contours
 * @return      index of contour in array of all contoursor -1 if point is not in any of them
 */
int inContours(std::vector<std::vector<cv::Point>> *contours, cv::Size dimensions, cv::Point position) {
    for (int i = 0; i < (*contours).size(); ++i) {

        cv::Mat druseImage(dimensions, CV_8UC1, cv::Scalar(0));
        drawContours( druseImage, (*contours), i, cv::Scalar(255, 255, 255), CV_FILLED, 8);
        if((int) druseImage.at<uchar>(position.x, position.y) == 255){
            return i;
        }
    }
    return -1;
}

/* @brief       Handles user inputs while learning process is active.
 * @param       event is type of event, i.e. EVENT_LBUTTONDOWN, EVENT_RBUTTONDOWN ...
 * @param       x position in image
 * @param       y position in image
 * @param       data is pointer to structure which contain parameters of function, must be of CallBackParams type
 */
void CallBackFunc(int event, int x, int y, int, void* data) {
    if  ( event == cv::EVENT_LBUTTONDOWN ) {
        CallBackParams *params = (CallBackParams *) data;
        auto index = inContours(&(*params).contours, (*params).image.size(), {y, x});

        // if any object was selected and object was not already selected, store its index to array
        if(index > -1 && std::find((*params).selectedIndexes.begin(), (*params).selectedIndexes.end(), index) == (*params).selectedIndexes.end()) {
            (*params).selectedIndexes.push_back(index);

            // highlight selected contours in image
            cv::Mat image;
            (*params).image.copyTo(image);
            for (int i = 0; i < (*params).selectedIndexes.size(); ++i) {
                drawContours( image, (*params).contours, (*params).selectedIndexes[i], cv::Scalar(255, 255, 255), CV_FILLED, 8);
            }
            cv::imshow((*params).windowName, image);
            cv::waitKey(1);
        }
    } else if  ( event == cv::EVENT_RBUTTONDOWN ) {
        CallBackParams *params = (CallBackParams *) data;
        auto index = inContours(&(*params).contours, (*params).image.size(), {y, x});

        if(index > -1) {

            // if is clicked contour in array of selected contours, remove it from array
            auto it = std::find((*params).selectedIndexes.begin(), (*params).selectedIndexes.end(), index);

            if(it != (*params).selectedIndexes.end()){

                (*params).selectedIndexes.erase(it);
            }

            // highlight selected contours in image
            cv::Mat image;
            (*params).image.copyTo(image);
            for (int i = 0; i < (*params).selectedIndexes.size(); ++i) {
                drawContours( image, (*params).contours, (*params).selectedIndexes[i], cv::Scalar(255, 255, 255), CV_FILLED, 8);
            }
            cv::imshow((*params).windowName, image);
            cv::waitKey(1);
        }
    }
}

/* @brief       Computes features of contours, i.e. area, compactness, average boundary intensity ...
 * @param       contours is pointer to array of elements contours
 * @param       image, in which contours were found
 * @param       index of contour in array
 * @return      structure of features
 */
ContourFeatures getContourFeature(std::vector<std::vector<cv::Point>> contours, cv::Mat image, int index, double ratio = std::numeric_limits<double>::infinity()) {

    // because of energy computing we image with all contours filled
    cv::Mat contoursMask(image.size(), CV_8UC1, cv::Scalar(0, 0, 0));
    for (int i = 0; i < contours.size(); i++) {
        drawContours(contoursMask, contours, i, cv::Scalar(255, 255, 255), CV_FILLED, 8);
    }

    cv::Mat elementMask, elementContoursMask, elementImage, elementContoursImage;
    elementMask = cv::Mat::zeros(image.size(), CV_8UC1);
    elementContoursMask = cv::Mat::zeros(image.size(), CV_8UC1);
    elementImage = cv::Mat::zeros(image.size(), CV_8UC1);
    elementContoursImage = cv::Mat::zeros(image.size(), CV_8UC1);

    ContourFeatures features;

    // compute sum of all pixels in element
    cv::drawContours(elementMask, contours, index, cv::Scalar(255), CV_FILLED, 8);
    image.copyTo(elementImage, elementMask);
    features.area = cv::sum(elementImage)[0];

    // compute compactness of element
    features.compactness = pow(cv::arcLength(contours[index], true), 2);

    //compute average boundary intensity
    drawContours(elementContoursMask, contours, index, cv::Scalar(255), 1, 8);
    cv::Mat intensity;
    cv::cvtColor(image, intensity, CV_BGR2GRAY);
    intensity.copyTo(elementContoursImage, elementContoursMask);
    features.avgBoundaryIntensity = cv::mean(elementImage)[0];

    // compute maximum and minimum boundary intensity
    double maxBoundaryIntensity, minBoundaryIntensity;
    cv::minMaxLoc(elementContoursImage, &minBoundaryIntensity, &maxBoundaryIntensity);
    features.maxBoundaryIntensity = maxBoundaryIntensity;

    // compute mean hue, saturation and intensity of element
    cv::Mat druseHSV, druseIntensity, druseHSVChannels[3];
    cv::cvtColor(elementImage, druseHSV, CV_BGR2HSV);
    cv::split(druseHSV, druseHSVChannels);
    intensity.copyTo(druseIntensity, elementMask);
    features.meanHue = cv::mean(druseHSVChannels[0], elementMask)[0];
    features.meanSaturation = cv::mean(druseHSVChannels[0], elementMask)[0];
    features.meanIntensity = cv::mean(druseIntensity, elementMask)[0];

    // compute mean gradient magnitude
    cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y, grad;
    // gradient X & Y
    Sobel(druseIntensity, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    Sobel(druseIntensity, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    features.meanGradientMagnitude = cv::mean(grad, elementMask)[0];

    // compute energy of candidate region
    features.energy = cv::sum(druseIntensity)[0] / cv::sum(contoursMask)[0];

    // compute entropy = measure of randomness
    int histMax = 256;
    features.entropy = computeEntropy(intensity, histMax, elementMask);

    if (ratio == std::numeric_limits<double>::infinity()) {
        features.ratio = computeRatio({image.cols, image.rows});
    } else {
        features.ratio = ratio;
    }

    return features;
}

/* @brief       Computes features of selected contours from image and save it to file.
 * @param       output is path to file
 * @param       image, in which objects for selection are found
 * @param       type of object, allowed values are LEARN_DRUSE = 0, LEARN_EXUDATE = 1, LEARN_HEMORRHAGE = 2
 */
void learn(std::string output, cv::Mat image, int type) {

    std::vector<std::vector<cv::Point>> contours;
    if(type == LEARN_DRUSE) {
        contours = getDrusenContours(image);
    } else if(type == LEARN_EXUDATE) {
        contours = getExudatesContours(image);
    } else if(type == LEARN_HEMORRHAGE) {
        //TODO pridat volanie hladania obrysov hemoragii
    } else {
        throw std::invalid_argument("Invalid 'method' property value. Enabled only LEARN_DRUSE = 0, LEARN_EXUDATE = 1, LEARN_HEMORRHAGE = 2.");
    }

    std::cout << "Extraction of contours completed ... " << std::endl;

    // show image and set listener for selecting contours area
    std::string windowTitle = "Select true findings ( SELECT - right mouse button, DESELECT - left mouse button, CONTINUE - press any key ) ...";
    cv::namedWindow(windowTitle, cv::WINDOW_NORMAL );
    cv::resizeWindow(windowTitle, 500,500);
    cv::moveWindow(windowTitle, 200,200);
    for (int j = 0; j < contours.size(); ++j) {
        drawContours( image, contours, j, cv::Scalar(255, 255, 255), 1, 8);
    }
    cv::imshow(windowTitle, image);
    CallBackParams params = {contours, image, windowTitle};
    cv::setMouseCallback(windowTitle, CallBackFunc, &params);
    cv::waitKey(0);
    cv::destroyWindow(windowTitle);
    cv::waitKey(1);

    std::cout << "Saving features of selected object." << std::endl;

    // save features of selected elements to file
    std::ofstream file;
    file.open(output);

    for (std::vector<int>::const_iterator i = params.selectedIndexes.begin(); i != params.selectedIndexes.end(); ++i) {
        ContourFeatures druse = getContourFeature(contours, image, *i);

        file << druse.area << ",";
        file << druse.compactness << ",";
        file << druse.avgBoundaryIntensity << ",";
        file << druse.minBoundaryIntensity << ",";
        file << druse.maxBoundaryIntensity << ",";
        file << druse.meanHue << ",";
        file << druse.meanSaturation << ",";
        file << druse.meanIntensity << ",";
        file << druse.meanGradientMagnitude << ",";
        file << druse.energy << ",";
        file << druse.entropy << ",";
        file << druse.ratio << "\n";
    }
    file.close();

    std::cout << "Features were successfully saved to file [" << output << "]." << std::endl;

}

/* @brief       Computes first eccentricity of ellipse.
 * @param       rectangle which describes ellipse
 * @return      double value of eccentricity
 */
double getEccentricity(cv::RotatedRect rectangle) {
    if(rectangle.size.height > rectangle.size.width) {
        return (double) (std::sqrt((std::pow(rectangle.size.height, 2) - std::pow(rectangle.size.width, 2))) / (rectangle.size.height));
    } else {
        return (double) (std::sqrt((std::pow(rectangle.size.width, 2) - std::pow(rectangle.size.height, 2))) / (rectangle.size.width));

    }
}

/* @brief       Finds ellipse, which describes macula in retina.
 * @param       image to be processed
 * @param       ratio is optional parameter, if not specified, new is computed by image parameter
 * @return      rotated rectangle which describe macula ellipse or zero-sized rectangle if no macula is found
 */
cv::RotatedRect getMaculaEllipse(cv::Mat image, double ratio = std::numeric_limits<double>::infinity()) {

    if (ratio == std::numeric_limits<double>::infinity()) {
        ratio = computeRatio({image.cols, image.rows});
    }

    // macula appears most contrasted in red channel
    std::vector<cv::Mat> bgrChannel;
    split(image, bgrChannel);
    cv::Mat redChannel = bgrChannel[2];

    // remove noise details by applying blur filter
    int size = ((int) (30 * ratio) | 1);
    cv::blur(redChannel, redChannel, {size, size});

    // do multilevel thresholding, macula appears most between value 100 - 230
    std::vector<cv::RotatedRect> minEllipse;
    for (int k = 100; k < 230; ++k) {
        cv::Mat threshold_output;
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        // detect edges using Threshold
        threshold(redChannel, threshold_output, k, 255, cv::THRESH_BINARY);

        // to exclude ellipses of whole retina,flood fill background
        cv::floodFill(threshold_output, {0, 0}, 255);

        // find contours
        findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

        // for each contour find rotated rectangles and ellipses
        for (int i = 0; i < contours.size(); i++) {
            if (contours[i].size() > 5) {
                auto elem = fitEllipse(cv::Mat(contours[i]));

                // check if they are ellipses and not circles by eccentricity value
                if (getEccentricity(elem) > MAX_MACULA_ECCENTRICITY) {
                    minEllipse.push_back(elem);
                }
            }
        }
    }

    if (minEllipse.size() > 0) {

        // collect ellipse to group with similar centroid
        std::vector<cv::Point2f> centers = {minEllipse[0].center};
        std::vector<int> belong(minEllipse.size(), 0);
        int iteration = 0;
        bool change;

        do {
            change = false;
            for (int i = 0; i < minEllipse.size(); i++) {

                // search for the nearest group
                int best = belong[i];
                double bestDistance = cv::norm(minEllipse[i].center - centers[best]);
                for (int j = 0; j < centers.size(); ++j) {
                    auto dist = cv::norm(minEllipse[i].center - centers[j]);
                    if (dist < bestDistance) {
                        bestDistance = dist;
                        best = j;
                        change = true;
                    }
                }

                // if is centroid of nearest group still too far away, create new group with this poin
                if (cv::norm(minEllipse[i].center - centers[best]) > (ratio * MAX_CLUSTERING_DISTANCE)) {
                    centers.push_back(minEllipse[i].center);
                    belong[i] = centers.size() - 1;
                } else {
                    belong[i] = best;
                }
            }

            // get members of each group
            std::vector<std::vector<cv::Point2f>> members(belong.size());
            for (int l = 0; l < belong.size(); ++l) {
                members[belong[l]].push_back(minEllipse[l].center);
            }

            // for each group re-calculate its centroid
            for (int k = 0; k < centers.size(); ++k) {
                centers[k] = {0, 0};
                for (int i = 0; i < members[k].size(); i++) {
                    centers[k] += members[k][i];
                }
                centers[k].x /= members[k].size();
                centers[k].y /= members[k].size();
            }

        } while (change && iteration++ < MAX_CLUSTERING_ITERATIONS);

        // get ellipses of each final group and the most numerous group
        std::vector<std::vector<cv::RotatedRect>> members(belong.size());
        auto max = 0;
        for (int l = 0; l < belong.size(); ++l) {
            members[belong[l]].push_back(minEllipse[l]);
            if (members[belong[l]].size() > members[max].size()) {
                max = belong[l];
            }
        }

        // find the largest ellipse in the most numerous group
        auto largest = 0;
        for (int i = 0; i < members[max].size(); i++) {
            if (members[max][i].size.area() > members[max][largest].size.area()) { largest = i; }
        }

        return members[max][largest];
    } else {
        return cv::RotatedRect();
    }
}

/* @brief       Creates mask of macula.
 * @param       image to be processed
 * @param       reduction is optional parameter, define reduction of dimensions of foreground image
 * @param       ratio is optional parameter, if not specified, new is computed by image parameter
 * @return      matrix with same size as original image, zero element are macula elements
 */
cv::Mat getMaculaMask(cv::Mat image, int reduction, double ratio = std::numeric_limits<double>::infinity()) {

    if(ratio == std::numeric_limits<double>::infinity()){
        ratio = computeRatio({image.cols, image.rows});
    }

    cv::RotatedRect macula = getMaculaEllipse(image, ratio);

    cv::Mat mask(image.rows, image.cols, CV_8UC1, cv::Scalar(255, 255, 255));
    cv::ellipse( mask, macula, cv::Scalar(0, 0, 0), CV_FILLED, 8 );
    if(reduction > 0) {
        cv::ellipse( mask, macula, cv::Scalar(0, 0, 0), reduction, 8 );
    }

    return mask;
}

/* @brief       Creates mask of blood vessels in retina image. Gabor filter is used.
 * @param       image to be processed
 * @param       opticDiscMask is mask in which zero elements are optic disc elements, if empty == cv::Mat(), new is computed by image
 * @param       ratio is optional parameter, if not specified, new is computed by image
 * @return      matrix with same size as original image, zero element are blood vessels elements
 */
cv::Mat getBloodVesselsMask(cv::Mat image, cv::Mat opticDiscMask, double ratio = std::numeric_limits<double>::infinity()) {

    cv::Mat backgroundMask = getBackgroundMask(image, 0, 0);

    if (ratio == std::numeric_limits<double>::infinity()) {
        std::vector<std::vector<cv::Point>> contours;
        findContours(backgroundMask.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        cv::Rect boundRect = boundingRect(cv::Mat(contours[0]));
        ratio = computeRatio(boundRect.size(), {660, 660});
    }

    // blood vessels appears most contrasted in inverted green channel
    std::vector<cv::Mat> bgrChannel;
    split(image, bgrChannel);
    cv::Mat greenChannel = bgrChannel[1];
    cv::subtract(cv::Scalar::all(255), greenChannel, greenChannel);

    cv::Mat structureElem = getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(15 * ratio, 15 * ratio)), morfGreenChannel;
    morphologyEx(greenChannel, greenChannel, cv::MORPH_CLOSE, structureElem);

    cv::Mat greenChannel32F;
    greenChannel32F = cv::Mat::zeros(greenChannel.rows, greenChannel.cols, CV_32FC1);
    greenChannel.convertTo(greenChannel32F, CV_32F, 1.0 / 255, 0);

    cv::Mat gaborFilterResponse;
    gaborFilterResponse = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);

    // rotate gabor kernel to find vessels in all angles
    for (int i = 0; i < 180; i += 30) {
        //ratio = 1;
        // compute width and height of kernel, i.e.: "|" shape is smaller width and bigger height and for "__" vice versa
        int size = 17 * ratio, kernelWidth = size, kernelHeight = size;
        double divisor = 30 / ratio;
        if(i <= 90){
            kernelHeight -= i / divisor;
            kernelWidth -= 3 * ratio - (i / divisor);
        } else {
            kernelWidth -= (i % 90) / divisor;
            kernelHeight -= 3 * ratio - ((i % 90) / divisor);
        }

        cv::Mat kernel =  cv::getGaborKernel({kernelWidth, kernelHeight}, 4.5 * ratio, i / 180. * M_PI, 9.9 * ratio, 1.3, CV_PI * 2);
        cv::Mat response;

        // proceed 2D filter of image by generated gabor kernel
        cv::filter2D(greenChannel32F, response, CV_32F, kernel);
        response.convertTo(response, CV_8U, 255);

        // add actual gabor filter result to total
        addWeighted( gaborFilterResponse, 0.5, response, 0.5, 0.0, gaborFilterResponse);
    }

    cv::Mat gaborThresh;
    threshold(gaborFilterResponse, gaborThresh, 5, 255, cv::THRESH_BINARY_INV);

    // find center of optic disc
    if(opticDiscMask.empty()){
        opticDiscMask = getOpticDiscMask(image, 5, ratio);
    }
    cv::Mat mask, vessels;
    bitwise_and(gaborThresh, opticDiscMask, mask);
    cv::Mat distanceMask;
    bitwise_not(mask, distanceMask);
    distanceTransform(distanceMask, distanceMask, CV_DIST_L2, 3);
    double min, max;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(distanceMask, &min, &max, &min_loc, &max_loc);
    normalize(mask, vessels, 0, 50,  cv::NORM_MINMAX);

    // use just those lines / objects, which goes from the optic disc
    floodFill(vessels, max_loc, 255);
    threshold(vessels, vessels, 254, 255, cv::THRESH_BINARY_INV);
    bitwise_or(gaborThresh, vessels, vessels);

    // remove optic disc from mask
    bitwise_and(vessels, backgroundMask, vessels);

    return vessels;
}

/* @brief       Finds contours of blood vessels in retina image. Local variation and morfological reconstruction are used.
 * @param       image to be processed
 * @param       ratio is optional parameter, if not specified, new is computed by image
 * @return      vector of contours ( vector of points ) of hemorrhages
 */
std::vector<std::vector<cv::Point>> getHemorrhagesContours(cv::Mat image, double ratio = std::numeric_limits<double>::infinity()){

    cv::Mat backgroundMask = getBackgroundMask(image, 0, 0);

    if (ratio == std::numeric_limits<double>::infinity()) {
        std::vector<std::vector<cv::Point>> contours;
        findContours(backgroundMask.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        cv::Rect boundRect = boundingRect(cv::Mat(contours[0]));
        ratio = computeRatio(boundRect.size(), {660, 660});
    }

    // hemorrhages appears most contrasted in inverted green channel
    std::vector<cv::Mat> bgrChannel;
    split(image, bgrChannel);
    cv::Mat greenChannel = bgrChannel[1];

    // make background color value as mean color of retina
    int meanColor = mean(greenChannel, backgroundMask)[0];
    bitwise_not(backgroundMask, backgroundMask);
    cv::Mat background = cv::Mat(greenChannel.size(), CV_8UC1, meanColor);
    background.copyTo(greenChannel, backgroundMask);

    // smooth optic disc and other bright lesions
    cv::Mat structureElem = getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(S1 * ratio, S1 * ratio)), morfGreenChannel;
    morphologyEx(greenChannel, morfGreenChannel, cv::MORPH_OPEN, structureElem);

    // contrast enhancement to improve the contrast of lesions for easy detection using
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1, cv::Size(8 * ratio, 8 * ratio));
    clahe->apply(morfGreenChannel, morfGreenChannel);

    // invert image
    bitwise_not(backgroundMask, backgroundMask);
    cv::subtract(cv::Scalar::all(255), morfGreenChannel, morfGreenChannel);

    // thresholding local variation image to extract candidate region, A1 is chosen in a very tolerant manner
    cv::Mat localVariationImage = localVariation(morfGreenChannel, backgroundMask, 4 * ratio);
    normalize(localVariationImage, localVariationImage, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    int A1 = 10;
    threshold(localVariationImage, localVariationImage, A1, 255, cv::THRESH_BINARY );

    // remove vessels from candidate regions
    cv::Mat vesselsMask = getBloodVesselsMask(image, cv::Mat(), ratio);
    structureElem = getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(4 * ratio, 4 * ratio));
    erode(vesselsMask, vesselsMask, structureElem);
    cv::Mat candidates; candidates = cv::Mat::zeros(vesselsMask.size(), CV_8UC1);
    localVariationImage.copyTo(candidates, vesselsMask);

    // create mask of candidate regions where all contours are filled
    std::vector<std::vector<cv::Point>> contours;
    findContours(candidates, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    candidates = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    for( size_t i = 0; i< contours.size(); i++ ) {
        drawContours( candidates, contours, (int)i, cv::Scalar(255,255,255), CV_FILLED, 0);
    }
    cvtColor(candidates, candidates, cv::COLOR_BGR2GRAY );

    // remove noise and enlarge candidate regions in mask
    erode(candidates, candidates, getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(3 * ratio, 3 * ratio)));
    dilate(candidates, candidates, getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(6 * ratio, 6 * ratio)));

    // set all the candidate regions to 0 in the original image
    cv::Mat candidatesMask; candidatesMask = cv::Mat::ones(image.rows, image.cols, CV_8UC1);
    cv::bitwise_not(candidates, candidates);
    cv::subtract(cv::Scalar::all(255), greenChannel, greenChannel);
    greenChannel.copyTo(candidatesMask, candidates);

    // calculate the morphological reconstruction by dilation
    cv::Mat reconstructedCandidates = morfologicalReconstruction(greenChannel, candidatesMask, structureElem, 0);

    // apply a threshold operation to the difference between the original image and the reconstructed image
    cv::Mat hemorrhages;
    subtract(greenChannel, reconstructedCandidates, hemorrhages);
    int A2 = 5;
    threshold(hemorrhages, hemorrhages, A2, 255, cv::THRESH_BINARY );

    findContours(hemorrhages, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    return contours;
}


int main( int argc, char** argv ) {

    cv::Mat image;
    image = cv::imread(argv[1], 1);

    if(!image.data ) {
        std::cout << "WARNING: No image data to process ..." << std::endl;
        return -1;
    }

    // make black borders of image so background mask can be extracted even if image contain cutout of retina
    copyMakeBorder(image, image, 50, 50, 50, 50, cv::BORDER_CONSTANT, 0);
    showImage("image", image);

    auto contours = getHemorrhagesContours(image);
    for(size_t i = 0; i< contours.size(); i++ ) {
        drawContours(image, contours, (int)i, cv::Scalar(87, 165, 154), 1, 1);
    }

    showImage("hemorrhages", image);

    cv::waitKey(0);
    return 0;
}