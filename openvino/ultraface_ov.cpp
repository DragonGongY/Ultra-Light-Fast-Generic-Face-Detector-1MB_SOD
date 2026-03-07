#include "ultraface_ov.h"
#include <fstream>
#include <algorithm>
#include <numeric>

UltrafaceDetector::UltrafaceDetector(const std::string& model_path,
                           float threshold,
                           float iou_threshold,
                           const std::string& device)
{
    prob_threshold = threshold;
    this->iou_threshold = iou_threshold;

    core = ov::Core();

    compiled_model = core.compile_model(
        model_path,
        device,
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
        ov::num_streams(4)
    );

    infer_request = compiled_model.create_infer_request();

    auto inputs = compiled_model.inputs();
    if (inputs.empty()) {
        throw std::runtime_error("Model has no inputs");
    }

    input_name = inputs[0].get_any_name();

    auto input_shape = inputs[0].get_shape();

    input_height = input_shape[2];
    input_width  = input_shape[3];

    auto outputs = compiled_model.outputs();

    for (const auto& output : outputs)
        output_names.push_back(output.get_any_name());
}

std::vector<Detection> UltrafaceDetector::detect(const cv::Mat& orig_image)
{
    cv::Mat image;

    cv::resize(orig_image, image, cv::Size(input_width, input_height));

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    image.convertTo(image, CV_32F, 1.0/128.0, -127.0/128.0);

    cv::Mat blob = cv::dnn::blobFromImage(image);

    ov::Shape input_shape = {1,3,(size_t)input_height,(size_t)input_width};

    ov::Tensor input_tensor(
        ov::element::f32,
        input_shape,
        blob.ptr<float>()
    );

    infer_request.set_tensor(input_name, input_tensor);

    infer_request.infer();

    if (output_names.empty())
        return {};

    ov::Tensor confidences_tensor;
    ov::Tensor boxes_tensor;

    if (output_names.size() >= 2)
    {
        confidences_tensor = infer_request.get_tensor(output_names[0]);
        boxes_tensor = infer_request.get_tensor(output_names[1]);
    }
    else
    {
        boxes_tensor = infer_request.get_tensor(output_names[0]);
        confidences_tensor = boxes_tensor;
    }

    return predict(
        orig_image.cols,
        orig_image.rows,
        confidences_tensor,
        boxes_tensor
    );
}

std::vector<Detection> UltrafaceDetector::predict(int width, int height, const ov::Tensor& confidences_tensor, const ov::Tensor& boxes_tensor)
{
    std::vector<Detection> result;

    auto boxes_shape = boxes_tensor.get_shape();

    if (boxes_shape.size() == 3 && boxes_shape[2] >= 6) {
        std::cout << "Processing single output format [batch, boxes, 6+]" << std::endl;
        int num_boxes = boxes_shape[1];
        int num_features = boxes_shape[2];
        const float* data = boxes_tensor.data<float>();

        for (int i = 0; i < num_boxes; i++) {
            int idx = i * num_features;
            float x1 = data[idx + 0] * width;
            float y1 = data[idx + 1] * height;
            float x2 = data[idx + 2] * width;
            float y2 = data[idx + 3] * height;
            float score = data[idx + 4];
            int class_id = (int)data[idx + 5];

            if (score < prob_threshold)
                continue;

            Detection det;
            det.box = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
            det.label = class_id;
            det.score = score;
            result.push_back(det);
        }
    } else {
        std::cout << "Processing dual output format" << std::endl;
        int num_boxes = boxes_shape[1];
        auto conf_shape = confidences_tensor.get_shape();
        int num_classes = conf_shape[2];

        const float* conf = confidences_tensor.data<float>();
        const float* box = boxes_tensor.data<float>();

        for (int class_index = 1; class_index < num_classes; class_index++)
        {
            std::vector<cv::Rect> class_boxes;
            std::vector<float> class_scores;

            for (int i = 0; i < num_boxes; i++)
            {
                float score = conf[i*num_classes + class_index];

                if (score < prob_threshold)
                    continue;

                float x1 = box[i*4+0] * width;
                float y1 = box[i*4+1] * height;
                float x2 = box[i*4+2] * width;
                float y2 = box[i*4+3] * height;

                cv::Rect r(cv::Point(x1,y1), cv::Point(x2,y2));

                class_boxes.push_back(r);
                class_scores.push_back(score);
            }

            if(class_boxes.empty())
                continue;

            std::vector<int> keep = hard_nms(class_boxes, class_scores);

            for(int id : keep)
            {
                Detection det;
                det.box = class_boxes[id];
                det.label = class_index;
                det.score = class_scores[id];
                result.push_back(det);
            }
        }
    }

    return result;
}

std::vector<int> UltrafaceDetector::hard_nms(
        const std::vector<cv::Rect>& boxes,
        const std::vector<float>& scores)
{
    std::vector<int> order(scores.size());

    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(),
        [&](int a,int b){return scores[a]>scores[b];});

    std::vector<int> keep;

    while(!order.empty())
    {
        int idx = order[0];

        keep.push_back(idx);

        std::vector<int> tmp;

        for(size_t i=1;i<order.size();i++)
        {
            if(iou(boxes[idx], boxes[order[i]]) <= iou_threshold)
                tmp.push_back(order[i]);
        }

        order = tmp;
    }

    return keep;
}

float UltrafaceDetector::iou(const cv::Rect& a,const cv::Rect& b)
{
    int xx1 = std::max(a.x,b.x);
    int yy1 = std::max(a.y,b.y);
    int xx2 = std::min(a.x+a.width,b.x+b.width);
    int yy2 = std::min(a.y+a.height,b.y+b.height);

    int w = std::max(0,xx2-xx1);
    int h = std::max(0,yy2-yy1);

    float inter = w*h;
    float uni = a.area()+b.area()-inter;

    return inter/uni;
}
