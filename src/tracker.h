#ifndef DETECTOR_CLASS_H_INCLUDED
#define DETECTOR_CLASS_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>

struct Object {
    int id;

    cv::Rect rect;
    cv::Scalar color;
    bool updated;
    std::list <cv::Point> history;
    cv::Point speed;
};

class ForegroundSubtractor {
public:
    ForegroundSubtractor (cv::BackgroundSubtractor *bg);
    ~ForegroundSubtractor ();

    cv::Mat get_foreground ();

    inline void ApplyMask (const cv::Mat &mask);
    inline void SubstractForeground (const cv::Mat &input);

    inline void Filtering ();

private:
    cv::BackgroundSubtractor *bg_;
    cv::Mat_ <uint8_t> foreground_;
};

class Tracker {
public:
    Tracker (cv::BackgroundSubtractor *bg);
    ~Tracker ();

    void SubstractBackground (cv::Mat &input, cv::Mat &mask);
    void DetectBlobs (const cv::Mat &input, cv::Size min_size = cv::Size (3, 3));
    void MatchObject (float minDist = 30);

    void DrawObjects (const cv::Mat &currentFrame, std::string name_of_window);
    void DisplayBlobs (const cv::Mat &currentFrame, std::string name_of_window) const;

private:
    void UpdateObjects (float minDist);
    void CreateObjects ();
    void EraseLostObjects ();

    ForegroundSubtractor *substractor_;

    std::list <Object> objects_;
    std::vector <cv::Rect> blobs_;

    cv::Mat_ <uint8_t> foreground_;
    int curId_;
};

#endif // DETECTOR_CLASS_H_INCLUDED
