#include "tools.hpp"

void grad_a(const cv::Mat& img,cv::Mat& dx,cv::Mat& dy)
{
    cv::Mat kernelx = (cv::Mat_<float>(1,3)<<-0.5, 0, 0.5);
    cv::Mat kernely = (cv::Mat_<float>(3,1)<<-0.5, 0, 0.5);
    cv::filter2D(img, dx, -1, kernelx);
    dx.col(0)=img.col(1)-img.col(0);
    dx.col(dx.cols-1)=img.col(img.cols-1)-img.col(img.cols-2);
    cv::filter2D(img, dy, -1, kernely);
    dy.row(0)=img.row(1)-img.row(0);
    dy.row(dy.rows-1)=img.row(img.rows-1)-img.row(img.rows-2);
}
cv::Mat jacobian_a(const int width,const int height)
{
    cv::Mat jac_x=cv::Mat::zeros(cv::Size(1,height),CV_32F);
    for(int wi=1;wi<width;++wi)
    {
        cv::hconcat(jac_x,wi*cv::Mat::ones(cv::Size(1,height),CV_32F),jac_x);
    } 
    
    cv::Mat jac_y=cv::Mat::zeros(cv::Size(width,1),CV_32F);
    for(int hi=1;hi<height;++hi)
    {
        cv::vconcat(jac_y,hi*cv::Mat::ones(cv::Size(width,1),CV_32F),jac_y);
    }
    cv::Mat jac_zero=cv::Mat::zeros(cv::Size(width,height),CV_32F);
    cv::Mat jac_ones=cv::Mat::ones(cv::Size(width,height),CV_32F);
    cv::Mat dwdpup=jac_x;
    hconcat(dwdpup,jac_zero,dwdpup);
    hconcat(dwdpup,jac_y,dwdpup);
    hconcat(dwdpup,jac_zero,dwdpup);
    hconcat(dwdpup,jac_ones,dwdpup);
    hconcat(dwdpup,jac_zero,dwdpup);
    
    cv::Mat dwdpdown=jac_zero;
    hconcat(dwdpdown,jac_x,dwdpdown);
    hconcat(dwdpdown,jac_zero,dwdpdown);
    hconcat(dwdpdown,jac_y,dwdpdown);
    hconcat(dwdpdown,jac_zero,dwdpdown);
    hconcat(dwdpdown,jac_ones,dwdpdown);
    
    cv::Mat dwdp=dwdpup;
    vconcat(dwdp,dwdpdown,dwdp);
    return dwdp;
}
cv::Mat warp_a(const cv::Mat& img,const cv::Matx23f& warp_p,
               const std::vector<cv::Point2f>& templt_pts)
{
    cv::Matx13f tmp={0,0,1};
    cv::Matx33f M;
    cv::vconcat(warp_p,tmp,M);
    M(0,0)+=1;
    M(1,1)+=1;
    cv::Mat wimg=quadtobox(img,templt_pts,M,"bilinear");
    return wimg;
}
void meshgrid(const int width,const int height,cv::Mat& outx,cv::Mat& outy)
{
    std::vector<float> x,y;
    for(int wi=1;wi<=width;++wi)
    {
        x.push_back(wi);
    }
    for(int hi=1;hi<=height;++hi)
    {
        y.push_back(hi);
    }
    cv::repeat(cv::Mat(x).t(),y.size(),1,outx);
    cv::repeat(cv::Mat(y),1,x.size(),outy);
}
cv::Mat quadtobox(const cv::Mat& img,const std::vector<cv::Point2f>& pts,
                  const cv::Matx33f& M,const std::string& ftype)
{
    //get min x and min y max x max y
    float min_x=pts[0].x,max_x=pts[0].x,min_y=pts[0].y,max_y=pts[0].y;
    for(int pi=1;pi<pts.size();++pi)
    {
        min_x=std::min(pts[pi].x,min_x);
        max_x=std::max(pts[pi].x,max_x);
        min_y=std::min(pts[pi].y,min_y);
        max_y=std::max(pts[pi].y,max_y);
    }
    cv::Mat xg,yg;
    meshgrid(max_x+1,max_y+1,xg,yg); 
    xg=xg.t();
    xg=xg.reshape(1,1); 
    yg=yg.t();
    yg=yg.reshape(1,1);
    cv::Mat xy;
    cv::vconcat(xg,yg,xy);
    cv::vconcat(xy,cv::Mat::ones(cv::Size(xy.cols,1),CV_32F),xy);

    cv::Mat uv=cv::Mat(M)*xy;
    //uv=uv(cv::Rect(0,0,uv.cols,2));
    uv=uv.reshape(1,(max_x+1)*3);
    cv::Mat xi=uv(cv::Rect(0,0,uv.cols,max_x+1)).t();
    cv::Mat yi=uv(cv::Rect(0,max_x+1,uv.cols,max_x+1)).t();
    
    cv::Mat wimg;
    cv::remap(img,wimg,xi,yi,cv::INTER_LINEAR);
    return wimg;
}
cv::Mat sd_images(const cv::Mat& dwdp,const cv::Mat& nIx,const cv::Mat& nIy,
                  const int np,const int h,const int w)
{
    cv::Mat vIdwdp(cv::Size(np*w,h),CV_32F);
    for(int ni=0;ni<np;++ni)
    {
        cv::Mat tx=nIx.mul(dwdp(cv::Rect((ni*w),0,w,h)));
        cv::Mat ty=nIy.mul(dwdp(cv::Rect((ni*w),h,w,dwdp.rows-h)));
        vIdwdp(cv::Rect(ni*w,0,w,h))=tx+ty;
    }
    return vIdwdp;
}
cv::Mat hessian_a(const cv::Mat& vIdwdp,const int np,const int w)
{
    cv::Mat hess=cv::Mat::zeros(cv::Size(np,np),CV_32F);
    for(int nci=0;nci<np;++nci)
    {
        cv::Mat h1=vIdwdp(cv::Rect((nci)*w,0,w,vIdwdp.rows));
        for(int nri=0;nri<np;++nri)
        {
            cv::Mat h2=vIdwdp(cv::Rect((nri)*w,0,w,vIdwdp.rows));
            hess.at<float>(nri,nci)=cv::sum(h1.mul(h2)).val[0];
        }
    }
    return hess;
}
cv::Mat sd_update(const cv::Mat& vIdwdp,const cv::Mat& err_img,const int np,
                  const int w)
{
    cv::Mat sd_delta_p=cv::Mat::zeros(cv::Size(1,np),CV_32F);
    for(int ni=0;ni<np;++ni)
    {
        cv::Mat h1=vIdwdp(cv::Rect((ni)*w,0,w,vIdwdp.rows));
        sd_delta_p.row(ni)=cv::sum(h1.mul(err_img));
    } 
    return sd_delta_p;
}
