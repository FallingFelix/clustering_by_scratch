#include <opencv2/opencv.hpp> // Include the main OpenCV header
#include <iostream>
#include <vector>

using namespace std;


int main(){
    //Vector of integers
    string imagePath = "filter1_img.jpg";
    // Read the image into a Mat object
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR); // IMREAD_COLOR loads as a 3-channel color image
    cv::Mat blurred_image;

    // Check if the image was loaded successfully
    if (image.empty()) {
        cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    // Display the image (optional)

    // Matrix filter
    // (1/16) * 
    // [1 2 1; 
    // 2 4 2; 
    // 1 2 1]


    for(int y = 0; y < image.cols; y++){
      for(int x = 0; x < image.rows; x++){
        image.at<uchar>(y,x) = 100 - image.at<uchar>(y,x);
      }
    }


    // Apply the filter to the image
    cv::imshow("Loaded Image", image);
    cv::waitKey(0); // Wait indefinitely for a key press

    return 0;
}
