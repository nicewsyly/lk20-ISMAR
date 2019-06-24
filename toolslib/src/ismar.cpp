#include <iostream>
#include <opencv2/opencv.hpp>
#include "ismar.hpp"

cv::Matx<float,2,3> ismarupdate_step(const cv::Matx<float,2,3>& warp_p,cv::Mat delta_p)
{
    delta_p=delta_p.reshape(1,3).t();
    cv::Matx<float,2,3> result=warp_p+cv::Matx23f(delta_p);
    return result;
}

std::vector<fit> ismar(const cv::Mat& firimg,const cv::Mat& secimg,
                           const std::vector<cv::Point2f> prevcontour,const cv::Matx23f& p_init)
{
    cv::Mat previmg,currimg;
    cv::cvtColor(firimg,previmg,cv::COLOR_BGR2GRAY);
    cv::cvtColor(secimg,currimg,cv::COLOR_BGR2GRAY);

    //cv::Mat drawimg=previmg.clone();
    //cv::rectangle(drawimg,conrect,cv::Scalar(255),4);
    //cv::imshow("img",drawimg);
    //cv::waitKey();

    //todo get fast points or grid points in contour
    cv::Mat con_pts=cv::Mat(get_conpts(previmg,prevcontour)).t();
    con_pts=con_pts.reshape(2,int(std::sqrt(con_pts.cols)));
    
    //todo get templ 
    previmg.convertTo(previmg,CV_32F); 
    currimg.convertTo(currimg,CV_32F); 
    //get gradient dx dy
    cv::Mat Idx,Idy;
    grad_a(currimg,Idx,Idy);

    std::vector<cv::Mat> xy;
    cv::split(con_pts,xy);
    cv::Mat templ;
    cv::remap(previmg,templ,xy[0],xy[1],cv::INTER_LINEAR);

    const int width=con_pts.cols;
    const int height=con_pts.rows; 
    cv::Mat dx,dy;
    ismargrad_a(previmg,con_pts,dx,dy);
    
    //todo evaluate jacobian
    cv::Mat dwdp=ismarjacobian_a(width,height);
    const int n_iters=50;
    std::vector<fit> fita(n_iters);
    const int np=6;
    cv::Matx23f warp_p=p_init;
    for(int ni=0;ni<n_iters;++ni)
    {
        //todo warp_a
        cv::Mat Iwxp=ismarwarp_a(currimg,warp_p,con_pts);
        cv::Mat err_img=templ-Iwxp;
        fita[ni].warp_p=warp_p;
        //mean image val
        cv::Scalar mean_scalar=cv::mean(err_img.mul(err_img));
        float mean_s=mean_scalar.val[0];
        fita[ni].rms_err=std::sqrt(mean_s);
        
        if(ni==n_iters-1)
        {
            break;
        }
        //get warp dx and dy
        cv::Matx13f tmp(0,0,1);
        cv::Matx33f M;
        cv::vconcat(warp_p,tmp,M);
        M(0,0)+=1;
        M(1,1)+=1;
        M(0,2)+=con_pts.at<cv::Vec2f>(0,0)[0];
        M(1,2)+=con_pts.at<cv::Vec2f>(0,0)[1];
        std::vector<cv::Mat> xy;
       
        cv::Mat pts=con_pts.reshape(2,1);
        cv::split(pts,xy);
        xy[0]-=con_pts.at<cv::Vec2f>(0,0)[0]-1;
        xy[1]-=con_pts.at<cv::Vec2f>(0,0)[1]-1; 
        cv::Mat xys;
        cv::vconcat(xy[0],xy[1],xys);
        cv::vconcat(xys,cv::Mat::ones(cv::Size(xys.cols,1),CV_32F),xys);
        cv::Mat uv=cv::Mat(M)*xys;
        
        cv::Mat nIx,nIy;
        cv::remap(Idx,nIx,uv.row(0),uv.row(1),cv::INTER_LINEAR);
        cv::remap(Idy,nIy,uv.row(0),uv.row(1),cv::INTER_LINEAR); 
        nIx=nIx.reshape(1,con_pts.rows);
        nIy=nIy.reshape(1,con_pts.rows);
        
        //todo steest descent image
        cv::Mat vIdwdp=ismarsd_images(dwdp,nIx,nIy,np,height,width);
        
        //todo hessian and iverse
        cv::Mat hess=ismarhessian_a(vIdwdp,np,width);
        cv::Mat hess_inv=hess.inv();
        cv::Mat sd_delta_p=ismarsd_update(vIdwdp,err_img,np,width);
        cv::Mat delta_p=hess_inv*sd_delta_p;
        warp_p=ismarupdate_step(warp_p,delta_p);
    }
    return fita;
}
