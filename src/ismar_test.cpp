#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include "affine_fa.hpp"
#include "affine_fc.hpp"
#include "affine_ia.hpp"
#include "affine_ic.hpp"
#include "affine_ic_d.hpp"
#include "affine_ic_lm.hpp"
#include "affine_ic_nt.hpp"
#include "affine_ic_nt_d.hpp"
#include "affine_ic_sd.hpp"
#include "ismar.hpp"
#include "tools.hpp"
#include <iostream>
#define DEBUG 1
#if DEBUG
    #define P(S,T) std::cout<<S<<T<<std::endl;
#else
    #define P(S,T) 
#endif

int main(int argc,char** argv)
{
    P("main","./optflow videopath");
    std::string videopath="../1.mov";//std::string(argv[1]);
    cv::VideoCapture video(videopath);
    if(!video.isOpened())
    {
        P("main","video file open failed");
        return 0;
    }
    cv::Mat templ;
    video.read(templ);
    cv::Rect templ_conrect=cv::Rect(73,242,100,100);
    std::vector<cv::Point2f> templ_contour={cv::Point2f(73,242),                        
				    cv::Point2f(172,242),    
				    cv::Point2f(172,341),
				    cv::Point2f(73,341)};
    //std::vector<cv::Point2f> templ_contour;
    //get_contour(templ,templ_contour);
    //cv::Rect templ_conrect=cv::boundingRect(templ_contour);
    
    
    cv::Mat frame;
    std::vector<cv::Point2f> contour(templ_contour.begin(),templ_contour.end());
    int cnt=0;    
    cv::Matx23f p_init=(cv::Matx23f(0,0,0,0,0,0));
    p_init(0,2)=-0.5;
    p_init(1,2)=-0.5;


    while(video.read(frame)&&!frame.empty()&&cnt++<25) 
    {
        std::cout<<"cnt "<<cnt<<std::endl;

        std::vector<fit> fita=ismar(templ,frame,templ_contour,p_init);     
        p_init=fita[fita.size()-1].warp_p;
        
        cv::Matx23f warp=fita[fita.size()-1].warp_p;
        cv::Matx33f M;
        cv::vconcat(warp,cv::Matx13f(0,0,1),M);
        M(0,0)+=1;
        M(1,1)+=1;
        std::vector<cv::Mat> xy;
        cv::split(cv::Mat(templ_contour).t(),xy);
        xy[0]=xy[0]-templ_contour[0].x;
        xy[1]=xy[1]-templ_contour[0].y;
        cv::Mat uv;
        cv::vconcat(xy[0],xy[1],uv);
        cv::vconcat(uv,cv::Mat::ones(cv::Size(uv.cols,1),CV_32F),uv);

        cv::Mat newxy=cv::Mat(M)*uv;
        cv::Mat newpts;
        newxy.row(0)+=templ_contour[0].x;
        newxy.row(1)+=templ_contour[0].y;
        mat2pts(cv::Matx34f(newxy),contour);

        std::cout<<"M "<< M<<std::endl; 
        std::cout<<"temp_contour "<<templ_contour<<std::endl;
        std::cout<<"curr_contour "<<contour<<std::endl;

        std::vector<cv::Point> draw_contour(contour.begin(),contour.end());
        std::cout<<"draw contour"<<draw_contour<<std::endl;
        cv::Mat drawimg=frame.clone();
        cv::polylines(drawimg,draw_contour,true,cv::Scalar(0,0,255),2); 
        cv::imshow("drawimg",drawimg);
        cv::waitKey();
         
        //templ=frame.clone();
        //templ_contour.assign(contour.begin(),contour.end());
    }
    return 0;
}
