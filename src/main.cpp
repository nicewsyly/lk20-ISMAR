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
#define DEBUG 0
#if DEBUG
    #define P(S,T) std::cout<<S<<T<<std::endl;
#else
    #define P(S,T) 
#endif

int main1(int argc,char** argv)
{
    P("main","./optflow videopath");
    std::string videopath="../1.mov";//std::string(argv[1]);
    cv::VideoCapture video(videopath);
    if(!video.isOpened())
    {
        P("main","video file open failed");
        return 0;
    }
    const long int cnt=video.get(cv::CAP_PROP_FRAME_COUNT);
    cv::Mat templ_;
    video.read(templ_);
    std::vector<cv::Point2f> templ_contour;
    //utils.hpp get_contour
    //get_contour(templ,templ_contour);
    //cv::cvtColor(templ_,templ_,cv::COLOR_BGR2GRAY);
    //templ_.convertTo(templ_,CV_32FC1,1.,0.);

    /*
    templ_contour={cv::Point2f(0,0),
                     cv::Point2f(0,399),
                     cv::Point2f(399,399),
                     cv::Point2f(399,0)};
    */
    cv::Rect templ_conrect=cv::Rect(73,242,100,100);   
    std::vector<cv::Point2f> templ_pts={cv::Point2f(templ_conrect.x,templ_conrect.y),
                                        cv::Point2f(templ_conrect.x+templ_conrect.width,templ_conrect.y),
                                        cv::Point2f(templ_conrect.x+templ_conrect.width,templ_conrect.y+templ_conrect.height),
                                        cv::Point2f(templ_conrect.x,templ_conrect.y+templ_conrect.height)};
    cv::Mat templ=templ_(templ_conrect).clone();
    //cv::Rect templ_conrect=cv::boundingRect(templ_contour);   
    //cv::Matx23f p_init =cv::Matx23f(0,0,72.5,0,0,242.5); 
    cv::Mat frame;
    cv::VideoWriter vw("../result.avi",cv::VideoWriter::fourcc('M','J','P','G'),15.0,templ_.size());
    if(!vw.isOpened())
    {
        std::cout<<"video writer file open failed"<<std::endl;
        return 0;
    }
    cv::Matx23f p_init=(cv::Matx23f(0,0,0,0,0,0));
    p_init(0,2)=-0.5;//conrect.x-1+0.5;
    p_init(1,2)=-0.5;//conrect.y-1+0.5;


    for(unsigned int ci=0;ci<cnt;++ci)
    {
        std::cout<<"cnt" <<ci<<std::endl;
        video.read(frame); 
        cv::Mat draw_img=frame.clone();
        //cv::cvtColor(frame,frame,cv::COLOR_BGR2GRAY);
        //frame.convertTo(frame,CV_32FC1,1.,0.); 
        //todo: affine_fa
        cv::Matx<float,2,3> warp_p;
        //std::vector<fit> fita=affine_fa(frame,templ,p_init,50,0,warp_p,templ_pts);
        std::vector<fit> fita=ismar(templ_,frame,templ_pts,p_init);
        //p_init=warp_p;
        p_init=fita[fita.size()-1].warp_p;
        std::cout<<ci<<" warp_p "<<p_init<<std::endl;
        cv::Matx33f M;
        cv::vconcat(p_init,cv::Matx13f(0,0,1),M);
        M(0,0)+=1;
        M(1,1)+=1;
        std::vector<cv::Point2f> curr_contour(templ_contour.size());
        P("M ",M);
        P("curr_contour mat size ",cv::Mat(curr_contour).size());
        cv::Matx34f templ_conmat=cv::Matx34f(templ_pts[0].x,templ_pts[1].x,templ_pts[2].x,templ_pts[3].x,
                                             templ_pts[0].y,templ_pts[1].y,templ_pts[2].y,templ_pts[3].y,1,1,1,1);
        //affine templ_contour;
        cv::Matx34f curr_conmat=M*templ_conmat;
        P("templ_conmat ",templ_conmat);
        P("curr_conmat ",curr_conmat);
        mat2pts(curr_conmat,curr_contour);
        std::vector<cv::Point> draw_contour(curr_contour.begin(),curr_contour.end());
        cv::polylines(draw_img,draw_contour,true,cv::Scalar(0,0,255));
        vw.write(draw_img);
        cv::imshow("frame",draw_img);
        cv::waitKey(10);
    }
    vw.release();
    
    return 0;
}
