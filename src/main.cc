#include "pcsc.hh"
#include "opencv2/opencv.hpp"
#include <iostream>

int main() {
    std::string imgpath = "../image/james.jpg"; // path of image.
    Image2D img(imgpath);   // Construct the Image2D object

    /// Compute and store Histogram of the image
    cv::Mat hist = img.ComputeHistogram();
    imwrite("../image/Histogram.jpg", hist);

    /// compute FFT of the original image and store the FFT result
    cv::Mat FFTImage = img.OutputFFTImage();    // Get the FFT image
    imwrite("../image/FFT.jpg", FFTImage);  // Store image

    /// Extract the image through Laplacian
    cv::Mat Contour = img.ContourExtraction();
    imwrite("../image/Contour.jpg", Contour);

    return 0;
}
