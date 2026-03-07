#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>
#include <string>

struct Detection
{
    cv::Rect box;
    int label;
    float score;
};

class UltrafaceDetector
{
public:

    UltrafaceDetector(const std::string& model_path,
                 float threshold = 0.7,
                 float iou_threshold = 0.3,
                 const std::string& device = "CPU");

    std::vector<Detection> detect(const cv::Mat& image);

private:

    std::vector<Detection> predict(int width, int height, const ov::Tensor& confidences, const ov::Tensor& boxes);

    std::vector<int> hard_nms(
            const std::vector<cv::Rect>& boxes,
            const std::vector<float>& scores);

    float iou(const cv::Rect& a, const cv::Rect& b);

private:

    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    std::string input_name;
    std::vector<std::string> output_names;
    std::vector<std::string> class_names{"ball"};

    float prob_threshold{0.7};
    float iou_threshold{0.3};

    int input_width{1280};
    int input_height{960};
};

#endif