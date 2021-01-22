#ifndef IMAGEPROCESSING
#define IMAGEPROCESSING

#include <iostream>
#include <opencv2/opencv.hpp>

class complex {
    /** complex class to store complex number and
     * overload some operators for complex number calulating.
     * Members:
     * re: real part of the complex number.
     * im: imaginary part of the com plex number.
    */
public:
    float re;
    float im;
    complex(float re=0.0, float im=0.0);
    ~complex(){};
    void operator= (complex ComNum);
    complex operator+ (complex ComNum);
    complex operator- (complex ComNum);
    complex operator* (complex ComNum);
    complex operator/ (complex ComNum);
};


class Image2D{
    /** This is a class to store and process image with 1 channel.
     * Members:
     * float **ppimage: store image by 2-dim array.
     * complex **FFTimg: store the result of FFT calculation. complex.
     * int rows: number of rows of the ppimage.
     * int cols: number of rows of the ppimage.
     *
     * Methods:
     * Image2D(std::string imagepath): load image through opencv, store data into ppimage.
     * int *GetRows(): Return #rows of the ppimage
     * int *GetCols(): Return #cols of the ppimage
     * float GetPixel(int i, int j): Return value of the ppimage pixel (i, j) (called when testing)
     * cv::Mat OutputImage(): convert data in ppimage to cv::Mat to show & write.
     * cv::Mat ComputeHistogram(): count the histogram of the ppimage.
     * void FFT2D(): calculate FFT of 2-dim array and store the result into FFTimg.
     * cv::Mat OutputFFTImage(): convert data in FFTimg to cv::Mat to show & write.
     * cv::Mat ContourExtraction(): extract the edge of the ppimage through Laplacian.
    */
public:
    Image2D(std::string imagepath);
    ~Image2D();
    int GetRows();
    int GetCols();
    float GetPixel(int i, int j);
    cv::Mat OutputImage();
    cv::Mat ComputeHistogram();
    void FFT2D();
    cv::Mat OutputFFTImage();
    cv::Mat ContourExtraction();
private:
    float **ppimage;
    complex **FFTimg;
    int rows;
    int cols;
};

#endif
