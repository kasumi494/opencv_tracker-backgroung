#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

#include "tracker.h"

// need -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video

using namespace cv;

int main (int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "No video file name" << std::endl;
        return -1;
    }

    std::string fileName = argv[1];
    VideoCapture input (fileName);

    if (!input.isOpened())
    {
        std::cerr << "Could not open file " << fileName << std::endl;
        return -1;
    }

    Tracker tracker (new BackgroundSubtractorMOG2);

    Mat frame;
    input >> frame;

    Mat_ <uint8_t> mask (frame.size (), uint8_t (255));
    if (argc == 3)
    {
        std::string maskName = argv[2];
        mask = imread (maskName, 0);
    }

    bool done = false;
    while (input.read (frame) && !done)
    {
        tracker.SubstractBackground (frame, mask);
        tracker.DetectBlobs (frame);
        tracker.DisplayBlobs (frame, "blobsFilteredImg");

        tracker.MatchObject ();

        // draw objects
        tracker.DrawObjects (frame, "Objects");

        switch ((char) waitKey (1))
        {
            case 27: case 'q':
                done = true;
                break;
            case ' ':
                waitKey ();
                break;
        }
    }

    std::cout << "Exit success" << std::endl;

    return 0;
}
