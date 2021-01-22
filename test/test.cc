#include <cmath>
#include <gtest/gtest.h>
#include "../src/pcsc.hh"

//float Image2DTest(std::string imgpath, int i, int j){
//    Image2D img = Image2D(imgpath);
//    img.GetPixel(i, j);
//}


/// Check if ppimage stored each img pixel correctly.
bool CheckStore(std::string imgpath){
    Image2D img = Image2D(imgpath);
    cv::Mat cvimage = cv::imread(imgpath, CV_LOAD_IMAGE_GRAYSCALE);
    for(int i = 0; i < img.GetRows(); i++){
        for(int j = 0; j < img.GetCols(); j++){
            float imgpixel = img.GetPixel(i, j);
            float cvimgpixel = (float)cvimage.data[i * cvimage.cols + j];
            if(abs(imgpixel - cvimgpixel) > 1e-5)
                return false;
        }
    }
    return true;
};



/// Check if Check if ppimage stored each img pixel correctly when constructing the Image2D object.
/// Use 3 different images.
TEST(Image2DTest1, Verify){
    ASSERT_TRUE(CheckStore("../image/james.jpg"));

};

TEST(Image2DTest2, Verify){
    ASSERT_TRUE(CheckStore("../image/frog.jpg"));
};

TEST(Image2DTest3, Verify){
    ASSERT_TRUE(CheckStore("../image/earth.jpg"));
};

/// Test correctness of operator overloading of class complex.
TEST(ComplexTest1, Verify){
    complex a = complex(4, 2);
    complex b = complex(3, 4);
    ASSERT_EQ(7, (a+b).re);
    ASSERT_EQ(6, (a+b).im);             /// (4+2i) + (3+4i) == 7+6i
    ASSERT_EQ(1, (a-b).re);
    ASSERT_EQ(-2, (a-b).im);            /// (4+2i) - (3+4i) == 1-2i
    ASSERT_EQ(4, (a*b).re);
    ASSERT_EQ(22, (a*b).im);            /// (4+2i) * (3+4i) == 4+22i
    ASSERT_NEAR(0.8, (a/b).re, 1e-6);
    ASSERT_NEAR(-0.4, (a/b).im, 1e-6);  /// (4+2i) / (3+4i) == 0.8+0.4i
}

TEST(ComplexTest2, Verify){
    complex a = complex(2, 6);
    complex b = complex(1, 3);
    ASSERT_EQ(3, (a+b).re);
    ASSERT_EQ(9, (a+b).im);             /// (2+6i) + (1+3i) == 3+9i
    ASSERT_EQ(1, (a-b).re);
    ASSERT_EQ(3, (a-b).im);             /// (2+6i) - (1+3i) == 1+3i
    ASSERT_EQ(-16, (a*b).re);
    ASSERT_EQ(12, (a*b).im);            /// (2+6i) * (1+3i) == -16+12i
    ASSERT_NEAR(2, (a/b).re, 1e-6);
    ASSERT_NEAR(0, (a/b).im, 1e-6);     /// (4+2i) / (3+4i) == 0.8+0.4i
}


TEST(ComplexTest3, Verify){
    complex a = complex(2, 3);
    complex b = complex(1, 1);
    ASSERT_EQ(3, (a+b).re);
    ASSERT_EQ(4, (a+b).im);             /// (2+3i) + (1+1i) == 3+4i
    ASSERT_EQ(1, (a-b).re);
    ASSERT_EQ(2, (a-b).im);             /// (2+3i) = (1+1i) == 1+2i
    ASSERT_EQ(-1, (a*b).re);
    ASSERT_EQ(5, (a*b).im);             /// (2+3i) * (1+1i) == -1+5i
    ASSERT_NEAR(2.5, (a/b).re, 1e-6);
    ASSERT_NEAR(0.5, (a/b).im, 1e-6);   /// (2+3i) / (3+1i) == 2.5+0.5i
}


int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
