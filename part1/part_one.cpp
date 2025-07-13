#include <opencv2/opencv.hpp> 
#include <iostream>
#include <vector>


using namespace std;
using namespace cv;

#define FIVE_MATRIX 5
#define THREE_MATRIX 3
#define THREE_MATRIX_SCALER (1.0f / 16.0f)
#define FIVE_MATRIX_SCALER (1.0f / 273.0f)

template <size_t N>

void matrixScaler(float (&kernal)[N][N], float scalar) {
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            kernal[i][j] *= scalar;
        }
    }
}

template <size_t N>
Mat applyFilterColor(const Mat& input, float (&kernel)[N][N]) {
    int pad = N / 2;

    // Split image into B, G, R channels
    vector<Mat> channels;
    split(input, channels); // input must be CV_8UC3

    vector<Mat> filtered_channels;

    for (int c = 0; c < 3; ++c) {
        Mat padded;
        copyMakeBorder(channels[c], padded, pad, pad, pad, pad, BORDER_CONSTANT, 0);
        Mat output(channels[c].size(), CV_8UC1);

        for (int y = pad; y < padded.rows - pad; ++y) {
            for (int x = pad; x < padded.cols - pad; ++x) {
                float sum = 0.0f;

                for (int ky = 0; ky < N; ++ky) {
                    for (int kx = 0; kx < N; ++kx) {
                        int iy = y - pad + ky;
                        int ix = x - pad + kx;
                        sum += kernel[ky][kx] * padded.at<uchar>(iy, ix);
                    }
                }

                int pixel_val = round(sum);
                pixel_val = min(255, max(0, pixel_val));
                output.at<uchar>(y - pad, x - pad) = static_cast<uchar>(pixel_val);
            }
        }

        filtered_channels.push_back(output);
    }

    // Merge filtered channels back into one image
    Mat merged;
    merge(filtered_channels, merged);
    return merged;
}

template <size_t N>
Mat applySobel(const Mat& input, int (&kernel)[N][N]) {
    int pad = N / 2;
    Mat padded;
    copyMakeBorder(input, padded, pad, pad, pad, pad, BORDER_CONSTANT, 0);
    Mat output(input.size(), CV_32F);

    for (int y = pad; y < padded.rows - pad; ++y) {
        for (int x = pad; x < padded.cols - pad; ++x) {
            float sum = 0.0f;
            for (int ky = 0; ky < N; ++ky) {
                for (int kx = 0; kx < N; ++kx) {
                    int iy = y - pad + ky;
                    int ix = x - pad + kx;
                    sum += kernel[ky][kx] * padded.at<uchar>(iy, ix);
                }
            }
            output.at<float>(y - pad, x - pad) = sum;
        }
    }

    return output;
}

Mat Sobelfilter(const Mat& original_image) {
    Mat gray;
    cvtColor(original_image, gray, COLOR_BGR2GRAY);

    // Define Sobel kernels
    int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int sobel_y[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    Mat gx = applySobel(gray, sobel_x);
    Mat gy = applySobel(gray, sobel_y);

    // Prevent overflow
    gx.convertTo(gx, CV_32F);
    gy.convertTo(gy, CV_32F);

    Mat magnitude;
    cv::magnitude(gx, gy, magnitude);

    // Normalize or scale safely
    Mat sobel_result;
    normalize(magnitude, sobel_result, 0, 255, NORM_MINMAX);
    sobel_result.convertTo(sobel_result, CV_8U);

    return sobel_result;
}

Mat dogFilters(const Mat& original_image) {
    Mat gray;
    cvtColor(original_image, gray, COLOR_BGR2GRAY);

    // Define Sobel kernels
    int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int sobel_y[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    Mat gx = applySobel(gray, sobel_x);
    Mat gy = applySobel(gray, sobel_y);

    // Prevent overflow
    gx.convertTo(gx, CV_32F);
    gy.convertTo(gy, CV_32F);

    // Optional display
    Mat gx_disp, gy_disp;
    convertScaleAbs(gx, gx_disp);
    convertScaleAbs(gy, gy_disp);
    imshow("gx Image", gx_disp);
    imshow("gy Image", gy_disp);

    // Compute gradient magnitude
    Mat magnitude;
    cv::magnitude(gx, gy, magnitude);

    // Normalize for display
    Mat sobel_result;
    normalize(magnitude, sobel_result, 0, 255, NORM_MINMAX);
    sobel_result.convertTo(sobel_result, CV_8U);

    return sobel_result;
}

int main() {
    string imagePath = "";

    for (int i = 0; i <= 1; ++i) {
        if (i == 0) {
            imagePath = "filter1_img.jpg";
        } else {
            imagePath = "filter2_img.jpg";
        }

        Mat image = imread(imagePath, IMREAD_COLOR);
        if (image.empty()) {
            cerr << "Error: Could not read the image.\n";
            continue;  // move on to next image
        }

        // Define kernels
        float kernel3x3[3][3] = {
            {1, 2, 1},
            {2, 4, 2},
            {1, 2, 1}
        };

        float kernel5x5[5][5] = {
            {1, 4, 7, 4, 1},
            {4,16,26,16, 4},
            {7,26,41,26, 7},
            {4,16,26,16, 4},
            {1, 4, 7, 4, 1}
        };

        matrixScaler(kernel3x3, 1.0f / 16.0f);
        matrixScaler(kernel5x5, 1.0f / 273.0f);

        Mat filtered3x3 = applyFilterColor(image, kernel3x3);
        Mat filtered5x5 = applyFilterColor(image, kernel5x5);
        Mat sobel_image = Sobelfilter(image);
        Mat dog_image= dogFilters(image);

        imshow("Original Image " + to_string(i), image);
        imshow("Filtered 3x3 Gaussian " + to_string(i), filtered3x3);
        imshow("Filtered 5x5 Gaussian " + to_string(i), filtered5x5);
        imshow("Sobel " + to_string(i), sobel_image);

    }

    waitKey(0);
    destroyAllWindows(); 
    return 0;
}
