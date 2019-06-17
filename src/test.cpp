#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include "tools.hpp"
#include "affine_fa.hpp"
#include "affine_fc.hpp"
#include "affine_ia.hpp"
#include <iostream>
#define DEBUG 1
#if DEBUG
    #define P(S,T) std::cout<<S<<T<<std::endl;
#else
    #define P(S,T) 
#endif

int main1(int argc,char** argv)
{
    cv::Mat a=(cv::Mat_<float>(1,2)<<1,4);
    std::cout<<"mean "<<cv::mean(a.mul(a))<<std::endl;

    cv::Mat img1=cv::imread("/home/parallels/images/easy/000.jpg");
    std::vector<cv::Point2f> templ_contour;
    cv::Rect templ_conrect(181,52,400,300);
    cv::imshow("templ",img1(templ_conrect));
    cv::waitKey();
    //get_contour(img1,templ_contour);
    //templ_contour={cv::Point2f(121, 35),cv::Point2f(516, 46),cv::Point2f(514, 305),cv::Point2f(129, 325)};
    cv::cvtColor(img1,img1,cv::COLOR_BGR2GRAY);
    img1.convertTo(img1,CV_32F,1.,0.);
    //cv::Rect templ_conrect=cv::boundingRect(templ_contour);

    cv::Matx23f p_init=cv::Matx23f(0,0,templ_conrect.x-1+0.5,0,0,templ_conrect.y-1+0.5);
    cv::Mat templ=img1(templ_conrect).clone();
    std::vector<cv::Point2f> templ_pts={cv::Point2f(0,0),cv::Point2f(templ_conrect.width-1,0),cv::Point2f(templ_conrect.width-1,templ_conrect.height-1),cv::Point2f(0,templ_conrect.height-1)};    

    cv::Mat img2=cv::imread("/home/parallels/images/easy/001.jpg");
    //cv::resize(img2,img2,cv::Size(img2.cols/2,img2.rows/2),cv::INTER_CUBIC);
    //cv::imshow("img2",img2);
    //cv::waitKey();
    cv::cvtColor(img2,img2,cv::COLOR_BGR2GRAY);
    img2.convertTo(img2,CV_32F,1.,0.);
    
    cv::Matx23f warp_p;
    std::vector<fit> fita=affine_fa(img2,templ,p_init,50,0,warp_p,templ_pts);
    for(int fi=0;fi<fita.size();++fi)
    {
        std::cout<<"warp_p "<<fita[fi].warp_p<<"error "<<fita[fi].rms_err<<std::endl;
    }
    
    

    
    return 0; 
}
