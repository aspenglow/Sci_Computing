#include "pcsc.hh"
#include <cmath>

#define PI 3.1415926
using namespace cv;

/// Constructor
complex::complex(float real, float imaginary){
    re = real;
    im = imaginary;
}

/// Operator overloading
void complex::operator=(complex ComNum){
    re = ComNum.re;
    im = ComNum.im;
    return;
}

complex complex::operator+(complex ComNum){
    complex result;
    result.re = re + ComNum.re;
    result.im = im + ComNum.im;
    return result;
}

complex complex::operator-(complex ComNum){
    complex result;
    result.re = re - ComNum.re;
    result.im = im - ComNum.im;
    return result;
}

complex complex::operator*(complex ComNum){
    complex result;
    result.re = re*ComNum.re - im*ComNum.im;
    result.im = re*ComNum.im + im*ComNum.re;
    return result;
}

complex complex::operator/(complex ComNum){
    complex result;
    result.re = (re*ComNum.re + im*ComNum.im) /
                (ComNum.re*ComNum.re + ComNum.im*ComNum.im);
    result.im = (im*ComNum.re - re*ComNum.im) /
                (ComNum.re*ComNum.re + ComNum.im*ComNum.im);
    return result;
}

/// Constructor
Image2D::Image2D(std::string imagepath) {
    Mat image = imread(imagepath, CV_LOAD_IMAGE_GRAYSCALE);
    if(!image.data){
        std::cerr << "no input image!" << std::endl;
    }
    rows = image.rows;
    cols = image.cols;

    /// FFTimg is square, each side length should be the largest between rows and cols
    int largest = (rows > cols ? rows : cols);
    /// dynamic allocation of 2-dim array.
    ppimage = new float*[rows];
    for(int i = 0; i < rows; i++){
        ppimage[i] = new float[cols];
        for(int j = 0; j < cols; j++){
            ppimage[i][j] = (float)image.data[i*cols+j];  // initialize
        }
    }
    FFTimg = new complex*[largest];
    for(int i = 0; i < largest; i++){
        FFTimg[i] = new complex[cols];
        for(int j = 0; j < largest; j++){
            FFTimg[i][j] = 0.0;
        }
    }
}

/// Destructor
Image2D::~Image2D(){
    for(int i = 0; i < rows; i++){
        delete[] ppimage[i];
        delete[] FFTimg[i];
    }
    delete[] FFTimg;
    delete[] ppimage;
}

/// Return shape (rows, cols) of the ppimage
int Image2D::GetRows(){
    return rows;
}

int Image2D::GetCols(){
    return cols;
}


/// Return value of the ppimage pixel (i, j)
float Image2D::GetPixel(int i, int j){
    return ppimage[i][j];
}

/// Convert data in ppimage to cv::Mat to show & write.
cv::Mat Image2D::OutputImage(){
    Mat output = Mat(rows, cols, CV_8U, Scalar::all(0));
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            output.at<uchar>(i,j) = (unsigned int)ppimage[i][j];
        }
    }
    return output;
}

/// Count the histogram of the ppimage.
cv::Mat Image2D::ComputeHistogram(){
    Mat hist = Mat(600, 700, CV_8U, Scalar::all(0)); // (600, 700): size of the histogram.
    /// Initialize histogram array.
    int graylevel[256];
    for(int i = 0; i < 256; i++){
        graylevel[i] = 0;
    }

    /// Count the histogram
    int TempPixel;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            TempPixel = (uint8_t)ppimage[i][j];
            graylevel[TempPixel]++;
        }
    }

    /// Scale the histogram
    int MaxGrayCount = 0;
    for(int i = 0; i < 256; i++){
        if(graylevel[i] > MaxGrayCount){
            MaxGrayCount = graylevel[i];
        }
    }

    float yscale = (float)(rows) / MaxGrayCount;
    float xscale = cols / 256;

    /// draw histogram with black rectangles.
    int x[257], y[256];
    for(int i = 0; i < 256; i++){
        y[i] = (int)(graylevel[i] * yscale);
    }
    x[0] = 0;
    for(int i = 1; i < 257; i++){
        x[i] = (int)(x[i-1] + xscale);
    }

    for(int i = 0; i < 256; i++){
        rectangle(hist, Point(x[i], y[i]), Point(x[i+1], 0), Scalar(255), -1 );
    }

    imshow("Histogram", hist);
    return hist;
}

/// 1-dim FFT, called by Image2D::FFT2D().
complex* fft(complex *A, int fft_nLen, int fft_M)
{
    int i;
    int lev, dist, p, t;
    complex B;
    complex *W = new complex[fft_nLen / 2];

    for (lev = 1; lev <= fft_M; lev++){
        dist = (int)pow(2, lev - 1);
        for (t = 0; t<dist; t++){
            p = t*(int)pow(2, fft_M - lev);
            W[p].re = (float)cos(2 * PI*p / fft_nLen);
            W[p].im = (float)(-1 * sin(2 * PI*p / fft_nLen));
            for (i = t; i<fft_nLen; i = i + (int)pow(2, lev)){
                B = A[i] + A[i + dist] * W[p];
                A[i + dist] = A[i] - A[i + dist] * W[p];
                A[i].re = B.re;
                A[i].im = B.im;
            }
        }
    }

    delete []W;
    return A;
}

/// Function to calculate i to expand image to ith power of 2 for FFT calculation.
/// Called by Image2D::FFT2D().
int CalculateL(int n){
    for(int i = 0; i < 32; i++){
        if((1<<i) >= n) return i;
    }
    return 0;
}

/// Called by Image2D::FFT2D().
int *reverse(int *b, int len, int M){
    int *a = new int[M];
    for (int i = 0; i<M; i++){
        a[i] = 0;
    }

    b[0] = 0;
    for (int i = 1; i<len; i++){
        int j = 0;
        while (a[j] != 0){
            a[j] = 0;
            j++;
        }
        a[j] = 1;
        b[i] = 0;
        for (j = 0; j<M; j++){
            b[i] = b[i] + a[j] * (int)pow(2, M - 1 - j);
        }
    }
    delete []a;
    return b;
}

/// Calculate FFT of 2-dim array and store the result into FFTimg.
/// Is called by the Image2D::OutputFFTImage().
void Image2D::FFT2D() {
    int R = CalculateL(rows);
    int C = CalculateL(cols);
    /// resize the image size to power of 2 (newrows, newcols)
    int newrows = 1;
    int newcols = 1;
    while (newrows < rows)
        newrows = newrows * 2;
    while (newcols < cols)
        newcols = newcols * 2;

    complex **A_in = new complex *[newrows];
    for (int i = 0; i < newrows; i++) {
        A_in[i] = new complex[newcols];
    }
    /// Up-left corner is the original image, the rest is padded by 0.
    for (int i = 0; i < newrows; i++) {
        for (int j = 0; j < newcols; j++) {
            if (i < rows && j < cols)
                A_in[i][j].re = ppimage[i][j];
            else
                A_in[i][j].re = 0.0;
            A_in[i][j].im = 0.0;
        }
    }

    /// Calculate FFT of the image
    /// By calculating 1-dim FFT of rows and cols respectively.
    complex *A = new complex[newcols];
    int *b = new int[newcols];
    b = reverse(b, newcols, C);
    for(int i = 0; i < newrows; i++){
        for (int j = 0; j < newcols; j++){
            A[j] = A_in[i][b[j]];
        }
        A = fft(A, newcols, C);
        for(int j = 0; j < newcols; j++){
            A_in[i][j] = A[j];
        }
    }

    delete []A;
    delete []b;

    A = new complex[newrows];
    b = new int[newrows];
    b = reverse(b, newrows, R);
    for(int i = 0; i < newcols; i++){
        for(int j = 0; j < newrows; j++){
            A[j] = A_in[b[j]][i];
        }
        A = fft(A, newrows, R);
        for(int j = 0; j < newrows; j++){
            A_in[j][i] = A[j];
        }
    }
    delete []A;
    delete []b;

    /// Upload the result of FFT into FFTimg.
    int largest = (rows > cols? rows: cols);
    for(int i = 0; i < largest; i++){
        for(int j = 0; j < largest; j++){
            FFTimg[i][j] = A_in[i][j];
        }
    }
    /// Release memory.
    for (int i = 0; i < newrows; i++) {
        delete []A_in[i];
    }
    delete []A_in;

    return;
}

/// convert data in FFTimg to cv::Mat to show & write.
cv::Mat Image2D::OutputFFTImage(){
    FFT2D();
    int largest = (rows > cols? rows: cols);
    Mat planes[] = { Mat::zeros(largest, largest, CV_32F), Mat::zeros(largest, largest, CV_32F) };
    Mat complexImg;
    merge(planes, 2, complexImg);
    for(int i = 0; i < largest; i++){
        for(int j = 0; j < largest; j++){
            complexImg.at<Vec2f>(i, j)[0] = (float)FFTimg[i][j].re;
            complexImg.at<Vec2f>(i, j)[1] = (float)FFTimg[i][j].im;
        }
    }

    /// if number of rows or columns is odd, crop
    split(complexImg, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat output = planes[0];
    output += Scalar::all(1);
    log(output, output);

    /// if the size is odd: remove the last row and col/
    output = output(Rect(0, 0, output.cols & (-2), output.rows & (-2)));

    int center = largest/2;
    Mat temp;
    Mat Quad0(output, Rect(0, 0, center, center));
    Mat Quad1(output, Rect(center, 0, center, center));
    Mat Quad2(output, Rect(0, center, center, center));
    Mat Quad3(output, Rect(center, center, center, center));
    /// Exchange data between 1st and 3rd quadrant.
    Quad3.copyTo(temp);
    Quad0.copyTo(Quad3);
    temp.copyTo(Quad0);
    /// Exchange data between 2nd and 4th quadrant.
    Quad2.copyTo(temp);
    Quad1.copyTo(Quad2);
    temp.copyTo(Quad1);

    normalize(output, output, 0, 1, CV_MINMAX);
    imshow("spectrum magnitude", output);
    output.convertTo(output, CV_8UC1, 255);
    return output;
}

/// Extract the edge of the ppimage through Laplacian.
cv::Mat Image2D::ContourExtraction(){
    /// Laplacian. can choose other operators by changing w
    float w[3][3] = {{0, -1, 0},
                   {-1, 4, -1},
                   {0, -1, 0}};
    /// Initialization
    Mat output =  Mat::zeros(rows, cols, CV_8U);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            float result = 0.0;
            /// Boundary conditions
            if(i == 0 || j == 0 || i == rows-1 || j == cols-1){
                result = ppimage[i][j];
            }
            else{ /// Laplacian
                result += w[0][0]*ppimage[i-1][j-1] + w[0][1]*ppimage[i-1][j] + w[0][2]*ppimage[i-1][j+1];
                result += w[1][0]*ppimage[i][j-1]   + w[1][1]*ppimage[i][j]   + w[1][2]*ppimage[i][j+1];
                result += w[2][0]*ppimage[i+1][j-1] + w[2][1]*ppimage[i+1][j] + w[2][2]*ppimage[i+1][j+1];
            }
            output.data[(i) * cols + (j)] = (unsigned int)result;
        }
    }
    imshow("contour extraction", output);
    waitKey();
    system("pause");
    return output;
}

