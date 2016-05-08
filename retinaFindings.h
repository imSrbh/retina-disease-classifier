//
// Created by janko on 8.5.2016.
//

#ifndef RETINAFINDINGS_H
#define RETINAFINDINGS_H

#include <iostream>
#include <vector>

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

const int MIN_FINDING_AREA = 20;
const int LEARN_DRUSE = 0;
const int LEARN_EXUDATE = 1;
const int LEARN_HEMORRHAGE = 2;

const std::string DB_FILE_HEADER_1 = "status, area, compactness, avg. boundary intensity, min. boundary intensity, max. boundary intensity, mean hue, mean saturation, mean intensity, mean gradient magnitude, energy, entropy, ratio";
const std::string DB_FILE_HEADER_2 = "status, area, eccentricity, perimeter, compactness, aspect ratio, mean of GC, std. dev. of GC, mean of CEGC, std. dev. of CEGC, mean gradient magnitude, neighbour mean gradient magnitude, mean hue, mean saturation, mean value, std. dev. hue, std. dev. saturation, std. dev. value, energy, entropy";

const std::string DRUSE_MODEL_NAME = "druse.xml";
const std::string EXUDATE_MODEL_NAME = "exudate.xml";
const std::string HEMORRHAGE_MODEL_NAME = "hemorrhage.xml";

// include sigma, theta, lambda, gamma, psi, kernel size
const std::vector<std::vector<double>> GABOR_FILTER_BANK = {
        {5,   0, 8,  1,     0, 70},
        {2,   0, 15, 0.7,   0, 70},
        {10,  0, 6,  0,     0, 70},
        {20,  0, 7,  0.15,  0, 30},
        {1.6, 0, 16, 0.16,  0, 30} };

/*  @brief  Structure that encapsulates all attributes that are necessary in callbackFunction for learning.
 */
struct CallBackParams {
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat image;
    std::string windowName;
    std::vector<int> positive;
    std::vector<int> negative;
};


/*  @brief     Creates structuring element in shape of hexagon where object image points are set to 255 and others are set to 0.
 *  @param     size is the length of one hexagon edge
 *  @return    matrix of hexagonal element
 */
cv::Mat getHexagonalStructuringElement(int size);

/*  @brief      Executes reconstruction of image by morfological operation while image before and after operation are not same.
 *  @param      mask of skipped image points, its non-zero elements indicate which matrix elements need to be processed
 *  @param      toReconstruct is matrix which is going to be processed
 *  @param      structureElem is matrix element by which morfological operation is computed
 *  @param      type of morfological operation: 0 = dilatation, other = erosion)
 *  @return     reconstructed image
 */
cv::Mat morfologicalReconstruction(cv::Mat, cv::Mat, cv::Mat, int);

/*  @brief      Filters contours by their area.
 */
void removeSmallAreaContours(std::vector<std::vector<cv::Point>>*, double);

/*  @brief      Computes ratio of loaded image and images used to test functions.
 *  @param      original are dimensions of loaded image
 *  @param      pattern are default dimensions of test image
 *  @return     double value of ratio
 */
double computeRatio(cv::Size, cv::Size);

/*  @brief      Finds retina image background mask by local variance of image or by contours of image
 *  @param      image to be processed
 *  @param      method - allowed values are BG_MASK_BY_CONTOURS = 0, BG_MASK_BY_LOCAL_VARIANCE = 1
 *  @param      reduction is optional parameter, define reduction of dimensions of foreground image
 *  @param      ratio is optional parameter, if not specified, new is computed by image parameter
 *  @return     matrix with same size as original image, zero element are background elements
 */
cv::Mat getBackgroundMask(cv::Mat, int, int);


/*  @brief      Compute correction of shades at image.
 *  @param      image to be processed
 *  @param      ratio is optional parameter, if not specified, new is computed by image parameter
 *  @return     matrix of shade corrected image
 */
cv::Mat shadeCorrection(cv::Mat, double = std::numeric_limits<double>::infinity());

/*  @brief      Computes standard local variation of matrix of input image.
 *  @param      image to be processed
 *  @param      mask of skipped image points, its non-zero elements indicate which matrix elements need to be processed
 *  @param      kernelSize define size of structuring element used for computing
 *  @param      max contains coordinates of point with the highest value of local variation
 *  @return     matrix of image local variation
 */
cv::Mat localVariation(cv::Mat, cv::Mat, int, cv::Point*);

/*  @brief      Finds optic disc center by standard local variation.
 *  @param      image to be processed
 *  @param      subImageSize define size of sub-image, which is used for improving center position of optic disc by distance tranformation
 *  @param      ratio is optional parameter, if not specified, new is computed by image parameter
 *  @return     point which points to optic disc center
 */
cv::Point getOpticDiscCenter(cv::Mat, int, double = std::numeric_limits<double>::infinity());

/*  @brief      Finds optic disc contours by watershed transformation.
 *  @param      image to be processed
 *  @param      ratio is optional parameter, if not specified, new is computed by image parameter
 *  @return     vector of contours ( vector of points ) of optic disc
 */
std::vector<std::vector<cv::Point>> getOpticDiscContours(cv::Mat, double = std::numeric_limits<double>::infinity());

/* @brief       Creates mask of optic disc.
 * @param       image to be processed
 * @param       reduction is optional parameter, define reduction of dimensions of foreground image
 * @param       ratio is optional parameter, if not specified, new is computed by image parameter
 * @return      matrix with same size as original image, zero element are optic disc elements
 */
cv::Mat getOpticDiscMask(cv::Mat, int, double = std::numeric_limits<double>::infinity());

/*  @brief      Computes histogram equalization of intensity channel.
 *  @param      image to be processed
 *  @return     image with equalized intensity
 */
cv::Mat equalizeIntensity(cv::Mat);

/*  @brief      Computes adaptive histogram equalization of color image.
 *  @param      image to be processed
 *  @param      clipLimit is threshold for contrast limiting
 *  @param      tileGridSize is size of grid for histogram equalization
 */
cv::Mat clahe(cv::Mat, double, cv::Size);

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
cv::Mat getGaborKernelImaginary(cv::Size, double, double, double, double, double, int = CV_64F);

/* @brief       Finds exudates contours in retina image.
 * @param       image to be processed
 * @param       opticDiscMask is mask of pixels where is optic disc, if it is empty ( all elemets of matrix are zeros), new one is computed in image
 * @param       ratio is optional parameter, if not specified, new is computed by image parameter
 * @return      vector of contours ( vector of points ) of exudates
 */
std::vector<std::vector<cv::Point>> getExudatesContours(cv::Mat, cv::Mat = cv::Mat(), double = std::numeric_limits<double>::infinity());

/* @brief       Finds contours of drusen in image of retina.
 * @param       image to be processed
 * @param       ratio is optional parameter, if not specified, new is computed by image parameter
 * @return      vector of contours ( vector of points ) of drusen
 */
std::vector<std::vector<cv::Point>> getDrusenContours(cv::Mat, double = std::numeric_limits<double>::infinity());

/* @brief       Computes entropy of image by histogram equalization.
 * @param       image to be processed
 * @param       histMax is maximum value of histogram
 * @param       mask of skipped image points, its non-zero elements indicate which matrix elements need to be processed
 * @return      entropy of image
 */
float computeEntropy(cv::Mat, int, cv::Mat);

/* @brief       Finds index of contour in which the point is.
 * @param       contours is pointer to array of elements contours
 * @param       dimensions are width and height of image from which contours are
 * @param       position is point in image, if out of contours
 * @return      index of contour in array of all contoursor -1 if point is not in any of them
 */
int inContours(std::vector<std::vector<cv::Point>>*, cv::Size, cv::Point);

/* @brief       Handles user inputs while learning process is active.
 * @param       event is type of event, i.e. EVENT_LBUTTONDOWN, EVENT_RBUTTONDOWN ...
 * @param       x position in image
 * @param       y position in image
 * @param       data is pointer to structure which contain parameters of function, must be of CallBackParams type
 */
void classificationCallBackFunc(int, int, int, int, void*);

/* @brief       Computes features of bright object in image (druse or exudate)
 * @param       contours is pointer to array of elements contours
 * @param       image, in which contours were found
 * @param       index of contour in array
 * @return      [area, compactness, average boundary intensity, maximum value of boundary intensity,
 *               mean hue, mean saturation, mean intensity, mean gradient magnitude, energy, entropy, ratio]
 */
std::vector<double> getBrightObjectFeature(std::vector<std::vector<cv::Point>>, cv::Mat*, int, double = std::numeric_limits<double>::infinity());


/* @brief       Computes first eccentricity of ellipse.
 * @param       rectangle which describes ellipse
 * @return      double value of eccentricity
 */
double getEccentricity(cv::RotatedRect);

/* @brief       Finds ellipse, which describes macula in retina.
 * @param       image to be processed
 * @param       ratio is optional parameter, if not specified, new is computed by image parameter
 * @return      rotated rectangle which describe macula ellipse or zero-sized rectangle if no macula is found
 */
cv::RotatedRect getMaculaEllipse(cv::Mat, double = std::numeric_limits<double>::infinity());

/* @brief       Creates mask of macula.
 * @param       image to be processed
 * @param       reduction is optional parameter, define reduction of dimensions of foreground image
 * @param       ratio is optional parameter, if not specified, new is computed by image parameter
 * @return      matrix with same size as original image, zero element are macula elements
 */
cv::Mat getMaculaMask(cv::Mat, int, double = std::numeric_limits<double>::infinity());

/* @brief       Creates mask of blood vessels in retina image. Gabor filter is used.
 * @param       image to be processed
 * @param       opticDiscMask is mask in which zero elements are optic disc elements, if empty == cv::Mat(), new is computed by image
 * @param       ratio is optional parameter, if not specified, new is computed by image
 * @return      matrix with same size as original image, zero element are blood vessels elements
 */
cv::Mat getBloodVesselsMask(cv::Mat, cv::Mat, double = std::numeric_limits<double>::infinity());

/* @brief       Finds contours of blood vessels in retina image. Local variation and morfological reconstruction are used.
 * @param       image to be processed
 * @param       ratio is optional parameter, if not specified, new is computed by image
 * @return      vector of contours ( vector of points ) of hemorrhages
 */
std::vector<std::vector<cv::Point>> getHemorrhagesContours(cv::Mat, double = std::numeric_limits<double>::infinity());

/* @brief       Computes features of hemorrhage contour
 * @param       contours is pointer to array of contours of hemorrhages
 * @param       image, in which contours were found
 * @param       index of contour in array
 * @return      [area, eccentricity, perimeter, compactness, aspect ratio,
 *              mean of candidate green channel, standart deviation of candidate green channel,
 *              mean of candidate contrast enhancement green channel, standart deviation of candidate contrast enhancement green channel,
 *              mean gradient magnitude, mean gradient of neighbor pixels,
 *              mean hue, mean saturation, mean value,
 *              standart deviation of hue, standart deviation of saturation, standart deviation of value,
 *              entropy, energy, ratio]
 */
std::vector<double> getHemorrhageFeature(std::vector<std::vector<cv::Point>>, cv::Mat*, int, double = std::numeric_limits<double>::infinity());

/* @brief       Checks if file already exists.
 * @param       fileName is path to file
 * @return      if exists true otherwise false
 */
bool fileExist(std::string);

/* @brief       Computes features of selected contours from image and save it to file.
 * @param       output is path to file
 * @param       image, in which objects for selection are found
 * @param       type of object, allowed values are LEARN_DRUSE = 0, LEARN_EXUDATE = 1, LEARN_HEMORRHAGE = 2
 */
void appendDatabase(cv::Mat, int type, std::string);

/* @brief       Splits string of float values separated by delimiter.
 * @param       s is string to be splited
 * @param       delimiter is substring in string which logically splits values
 * @param       values is empty vector and will be filled by splitted float values
 */
void split(const std::string&, char, std::vector<float>&);

/* @brief       Converts vector of float values to matrix.
 * @param       vec is input vector
 * @return      matrix of converted values
 */
cv::Mat convert(std::vector<std::vector<float>>);

/* @brief       Trains bayes classifier by input data and save generated model to file.
 * @param       input is system path to file in which is stored local database of retina findings and their classification as true/false positive
 * @param       output is system path to folder where statistical model will be saved as XML / YAML file
 * @param       type of object, allowed values are LEARN_DRUSE = 0, LEARN_EXUDATE = 1, LEARN_HEMORRHAGE = 2
 */
void trainClassifier(std::string, std::string, int);

#endif //RETINADISEASECLASIFIER_RETINAFINDINGS_H
