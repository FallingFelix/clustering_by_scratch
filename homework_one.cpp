#include <opencv2/opencv.hpp> // Include the main OpenCV header
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

int main(){
    float small_kernal[3][3] = {
        {1,2,1},
        {2,4,2},
        {1,2,1}
    };

    float large_kernal[5][5] = {
        {1,4,7,4,1},
        {4,16,26,15,4},
        {7,27,41,27,7},
        {4,16,26,15,4},
        {1,4,7,4,1}
    };

    matrixScaler(small_kernal, THREE_MATRIX_SCALER);
    matrixScaler(large_kernal, FIVE_MATRIX_SCALER);


    //Vector of integers
    string imagePath = "filter1_img.jpg";
    // Read the image into a Mat object
    Mat image = imread(imagePath, IMREAD_COLOR); // IMREAD_COLOR loads as a 3-channel color image
    Mat blurred_image;

    // Check if the image was loaded successfully
    if (image.empty()) {
        cerr << "Error: Could not read the image." << endl;
        return -1;
    }

    Mat padded;
    int padding = 1;

    copyMakeBorder(
        image, padded,
        padding, padding, padding, padding,
        BORDER_CONSTANT, Scalar(0) //pad with zeros
    );


    int inverted_value = 100;
    for(int y = 0; y < padded.rows; y++){
      for(int x = 0; x < padded.cols; x++){
        Vec3b &pixel = padded.at<Vec3b>(y,x);
        pixel[0] = inverted_value - pixel[0];
        pixel[1] = inverted_value - pixel[1];
        pixel[2] = inverted_value - pixel[2];
      }
    }


    // Apply the filter to the padded 
    imshow("Loaded Padded", padded);
    waitKey(0); // Wait indefinitely for a key press

    return 0;
}
