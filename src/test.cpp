#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include "tools.hpp"
#include "ismar.hpp"
#include <iostream>
#define DEBUG 1
#if DEBUG
    #define P(S,T) std::cout<<S<<T<<std::endl;
#else
    #define P(S,T) 
#endif

int main2(int argc,char** argv)
{
    cv::Mat img1=cv::imread("/home/parallels/images/hard/000.jpg");
    cv::Mat img2=cv::imread("/home/parallels/images/hard/001.jpg");
    std::vector<cv::Point2f> templ_contour;
    cv::Matx23f warp_p;
    //templ_contour={cv::Point2f(181,52),
    //               cv::Point2f(281,52),
    //               cv::Point2f(281,152),
    //               cv::Point2f(181,152)};
    //
    templ_contour={cv::Point2f(365,132),cv::Point2f(464,132),                    
                   cv::Point2f(464,231),cv::Point2f(365,231)};
    cv::Matx23f p_init=(cv::Matx23f(0,0,0,0,0,0));
    p_init(0,2)=-0.5;//conrect.x-1+0.5;
    p_init(1,2)=-0.5;//conrect.y-1+0.5;


    std::vector<fit> fita=ismar(img1,img2,templ_contour,p_init);

    for(int fi=0;fi<fita.size();++fi)
    {
        std::cout<<fi<<std::endl;
        std::cout<<"warp_p "<<fita[fi].warp_p<<"\nerror "<<fita[fi].rms_err<<std::endl;
    }
    
    

    
    return 0; 
}
