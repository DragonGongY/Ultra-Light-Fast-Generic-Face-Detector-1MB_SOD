#include <iostream>
#include <fstream>
#include "ultraface_ov.h"  

int main(int argc, char* argv[]) {
    std::cout << "Hello, World!" << std::endl;
    if(argc != 3)
    {
        std::cout<<"Usage: "<<argv[0]<<" <model_path> <video_path>\n";
        return -1;
    }

    std::string model = argv[1];
    UltrafaceDetector detector(model);

    cv::VideoCapture cap(argv[2]);

    if(!cap.isOpened())
    {
        std::cout<<"video open failed\n";
        return -1;
    }

    int sum = 0;

    while(true)
    {
        cv::Mat frame;
        cap >> frame;

        if(frame.empty())
        {
            std::cout<<"no img\n";
            break;
        }

        auto t1 = cv::getTickCount();

        auto detections = detector.detect(frame);

        auto t2 = cv::getTickCount();

        double cost =
            (t2 - t1)/cv::getTickFrequency();

        std::cout<<"cost time:"<<cost<<std::endl;

        for(auto& det : detections)
        {
            rectangle(frame, det.box, cv::Scalar(255,255,0),4);
        }

        sum += detections.size();

        cv::imshow("annotated", frame);

        if(cv::waitKey(1)=='q')
            break;
    }

    std::cout<<"sum:"<<sum<<std::endl;

    return 0;
}
