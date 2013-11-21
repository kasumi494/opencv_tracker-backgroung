#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#include "tracker.h"

using namespace cv;

ForegroundSubtractor::ForegroundSubtractor (BackgroundSubtractor *bg)
{
    bg_ = bg;
}

ForegroundSubtractor::~ForegroundSubtractor ()
{
    delete bg_;
}

cv::Mat ForegroundSubtractor::get_foreground ()
{
    return foreground_;
}

inline void ForegroundSubtractor::ApplyMask (const cv::Mat &mask)
{
    bitwise_and (foreground_, mask, foreground_);
}

inline void ForegroundSubtractor::SubstractForeground (const cv::Mat &input)
{
    bg_->operator () (input, foreground_);
}

inline void ForegroundSubtractor::Filtering ()
{
    erode  (foreground_, foreground_, Mat());
    dilate (foreground_, foreground_, Mat());
    dilate (foreground_, foreground_, Mat());

    //    imshow ("dilate2", foreground_);
}

///////////////////////////////////////////////////////////////////////////////

Tracker::Tracker (BackgroundSubtractor *bg)
{
    curId_ = 0;
    substractor_ = new ForegroundSubtractor (bg);
}

Tracker::~Tracker ()
{
    delete substractor_;
}

void Tracker::SubstractBackground (Mat &input, Mat &mask)
{
    substractor_->SubstractForeground (input);

    substractor_->ApplyMask (mask);
    substractor_->Filtering ();

    foreground_ = substractor_->get_foreground ();
}

void Tracker::DetectBlobs (const Mat &currentFrame, Size min_size)
{
    std::vector < std::vector <Point> > contours;
    findContours (foreground_, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    Mat contoursImg = currentFrame.clone ();
    drawContours (contoursImg, contours, -1, CV_RGB (0, 255, 0));
    imshow ("contours", contoursImg);

    /// All finding blobs
    for (int i = 0, lim = contours.size(); i < lim; ++i)
        blobs_.push_back (boundingRect (contours[i]));

 //   DisplayBlobs ("blobsImg");

    /// Erase bad blobs
    std::vector <Rect>::iterator it = blobs_.begin ();
    std::vector <Rect>::iterator it_end = blobs_.end ();

    for (; it != it_end; ++it)
        if (it->size().width < min_size.width || it->height < min_size.height)
            blobs_.erase (it);

    //DisplayBlobs ("blobsFilteredImg");
}

void Tracker::DisplayBlobs (const Mat &currentFrame, std::string name_of_window) const
{
    Mat image = currentFrame.clone();

    for (int i = 0, lim = blobs_.size(); i < lim; ++i)
        rectangle (image, blobs_[i], CV_RGB (0, 255, 0), 1);

    imshow (name_of_window, image);
}

void Tracker::DrawObjects (const Mat &currentFrame, std::string name_of_window)
{
    Mat objectImg = currentFrame.clone();

    std::list <Object>::iterator objI = objects_.begin();
    std::list <Object>::iterator objI_end = objects_.end();

    for(; objI != objI_end; ++objI)
    {
        std::list <Point>::const_iterator it = objI->history.begin();
        std::list <Point>::const_iterator it_end = objI->history.end();

        for (; it != it_end; ++it)
            circle (objectImg, *it, 1, objI->color, 1);

        rectangle (objectImg, objI->rect, objI->color, 1);

        Point center = Point (objI->rect.x + objI->rect.width / 2.0,
                              objI->rect.y + objI->rect.height / 2.0);

        line (objectImg, center, center + objI->speed * 3, CV_RGB(0,255,0), 2);
        putText (objectImg, format ("%d", objI->id), Point (objI->rect.x, objI->rect.y - 15),
                 FONT_HERSHEY_SIMPLEX, 1.0, objI->color, 2);
    }

    imshow (name_of_window, objectImg);
}

void Tracker::UpdateObjects (float minDist)
{
    std::list <Object>::iterator objI       = objects_.begin();
    std::list <Object>::iterator objI_end   = objects_.end();

    for (; objI != objI_end; ++objI)
    {
        Object& object = *objI;
        object.updated = false;

        float current_min_dist = minDist;

        std::vector <Rect>::iterator nearestBlob = blobs_.end();

        std::vector <Rect>::iterator blobI = blobs_.begin();
        std::vector <Rect>::iterator blobI_end = blobs_.end();

        for (; blobI != blobI_end; ++blobI)
        {
            float dist = norm (Point2f (object.rect.x, object.rect.y) -
                               Point2f (blobI->x, blobI->y));

            if (dist < current_min_dist)
            {
                current_min_dist = dist;
                nearestBlob = blobI;
            }
        }

        if (nearestBlob != blobs_.end())
        {
            object.speed = Point (nearestBlob->x - object.rect.x,
                                  nearestBlob->y - object.rect.y);

            object.rect = *nearestBlob;
            object.history.push_back (Point (object.rect.x + object.rect.width/2.0,
                                             object.rect.y + object.rect.height/2.0));
            object.updated = true;

            blobs_.erase (nearestBlob);
        }
    }
}

void Tracker::CreateObjects ()
{
    std::vector <Rect>::const_iterator it = blobs_.begin();
    std::vector <Rect>::const_iterator it_end = blobs_.end();

    for (; it != it_end; ++it)
    {
        Object newObject;
        newObject.rect = *it;
        newObject.id = curId_++;
        newObject.color = CV_RGB (rand() % 255, rand() % 255, rand() % 255);
        newObject.updated = true;

        objects_.push_back (newObject);
    }
}

void Tracker::EraseLostObjects ()
{
    std::list <Object>::iterator it = objects_.begin();
    std::list <Object>::iterator it_end = objects_.end();

    for (; it != it_end; )
    {
        if (!it->updated)   it = objects_.erase (it);
        else    ++it;
    }
}

void Tracker::MatchObject (float minDist)
{
    UpdateObjects (minDist);
    CreateObjects ();
    EraseLostObjects ();
}
