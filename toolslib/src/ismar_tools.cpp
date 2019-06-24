#include "ismar_tools.hpp"
#include <cmath>
#include "tools.hpp"

std::vector<cv::Point2f> get_conpts(const cv::Mat& img,
                                    const std::vector<cv::Point2f>& contour)
{
    std::vector<cv::Point2f> conpts;
    cv::Rect conrect=cv::boundingRect(contour);
    cv::Ptr<cv::FeatureDetector> trkdet=cv::FastFeatureDetector::create();
   //return conpts;
    //std::vector<cv::Point2f> conpts;
    //cv::Rect conrect=cv::boundingRect(contour);
    for(int ri=conrect.y;ri<conrect.y+conrect.height;ri+=1)
    {
        for(int ci=conrect.x;ci<conrect.x+conrect.width;ci+=1)
        {
            if(cv::pointPolygonTest(contour,cv::Point2f(ci,ri),false)>0)
                conpts.push_back(cv::Point2f(ci,ri));
        }
    }
    int sq=int(std::sqrt(conpts.size())); 
    conpts.resize(sq*sq);
    /*
    std::vector<cv::KeyPoint> detkps;
    cv::Mat conmat=img(conrect);
    trkdet->detect(conmat,detkps,cv::noArray()); 
    for(unsigned int di=0;di<detkps.size();++di)
    {
        if(pointPolygonTest(contour,detkps[di].pt+cv::Point2f(conrect.x,conrect.y),false)>0)
        {
            conpts.push_back(detkps[di].pt+cv::Point2f(conrect.x,conrect.y));
        }
    }
    std::cout<<"conpts size "<<conpts.size()<<std::endl;
 
    //conpts.resize(6400);
    int sq=int(std::sqrt(conpts.size())); 
    conpts.resize(sq*sq);
    */
    return conpts;
}
void ismargrad_a(const cv::Mat& img,const cv::Mat& con_pts,
                 cv::Mat& dx,cv::Mat& dy)
{
    //std::cout<<"into grad and conpts cols "<<con_pts.cols<<
               //" rows "<<con_pts.rows<<"type "<<con_pts.type()<<std::endl;
    dx.release();
    dy.release();
    dx=cv::Mat::zeros(con_pts.size(),CV_32F);
    dy=cv::Mat::zeros(con_pts.size(),CV_32F);
    const float* cdata=con_pts.ptr<float>(0);
    float* xdata=dx.ptr<float>(0);
    float* ydata=dy.ptr<float>(0); 
    const float* idata=img.ptr<float>(0);
    for(unsigned int ci=0;ci<con_pts.cols;++ci)
    {
        cv::Point pt=cv::Point(std::round(cdata[ci*2]),std::round(cdata[ci*2+1]));
        if(pt.x==0)
            xdata[ci]=int(img.ptr<float>(pt.y)[pt.x+1])-int(img.ptr<float>(pt.y)[pt.x]);
        else if(pt.x==img.cols-1)
            xdata[ci]=img.ptr<float>(pt.y)[pt.x]-img.ptr<float>(pt.y)[pt.x-1];
        else if(pt.y==0)
            ydata[ci]=img.ptr<float>(pt.y+1)[pt.x]-img.ptr<float>(pt.y)[pt.x];
        else if(pt.y==img.rows-1)
            ydata[ci]=img.ptr<float>(pt.y)[pt.x]-img.ptr<float>(pt.y-1)[pt.x];
        else
        {
            xdata[ci]=(img.ptr<float>(pt.y)[pt.x+1]-img.ptr<float>(pt.y)[pt.x-1])/2.;
            ydata[ci]=(img.ptr<float>(pt.y+1)[pt.x]-img.ptr<float>(pt.y-1)[pt.x])/2.;
        }
    }
    
    /*
    cv::Mat kernelx = (cv::Mat_<float>(1,3)<<-0.5, 0, 0.5);
    cv::Mat kernely = (cv::Mat_<float>(3,1)<<-0.5, 0, 0.5);
    cv::filter2D(img, dx, -1, kernelx);
    dx.col(0)=img.col(1)-img.col(0);
    dx.col(dx.cols-1)=img.col(img.cols-1)-img.col(img.cols-2);
    cv::filter2D(img, dy, -1, kernely);
    dy.row(0)=img.row(1)-img.row(0);
    dy.row(dy.rows-1)=img.row(img.rows-1)-img.row(img.rows-2);
    */
    
}
cv::Mat ismarjacobian_a(const int width,const int height)
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
cv::Mat ismarwarp_a(const cv::Mat& img,const cv::Matx23f& warp_p,
               const cv::Mat& con_pts)
{
    cv::Matx13f tmp={0,0,1};
    cv::Matx33f M;
    cv::vconcat(warp_p,tmp,M);
    M(0,0)+=1;
    M(1,1)+=1;
    M(0,2)+=con_pts.at<cv::Vec2f>(0,0)[0]+1;
    M(1,2)+=con_pts.at<cv::Vec2f>(0,0)[1]+1;
    cv::Mat wimg=ismarquadtobox(img,con_pts,M,"bilinear");
    return wimg.reshape(1,con_pts.rows);
}
void ismarmeshgrid(const int width,const int height,cv::Mat& outx,cv::Mat& outy)
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
cv::Mat ismarquadtobox(const cv::Mat& img,const cv::Mat& con_pts,
                  const cv::Matx33f& M,const std::string& ftype)
{
    //get min x and min y max x max y
    /*
    float min_x=pts[0].x,max_x=pts[0].x,min_y=pts[0].y,max_y=pts[0].y;
    for(int pi=1;pi<pts.size();++pi)
    {
        min_x=std::min(pts[pi].x,min_x);
        max_x=std::max(pts[pi].x,max_x);
        min_y=std::min(pts[pi].y,min_y);
        max_y=std::max(pts[pi].y,max_y);
    }
    cv::Mat xg,yg;
    ismarmeshgrid(max_x+1,max_y+1,xg,yg); 
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
    */ 
    cv::Mat pts=con_pts.reshape(2,1);
    std::vector<cv::Mat> xy;
    cv::split(pts,xy);
    cv::Mat xys;
    xy[0]-=con_pts.at<cv::Vec2f>(0,0)[0]-1;
    xy[1]-=con_pts.at<cv::Vec2f>(0,0)[1]-1;
    cv::vconcat(xy[0],xy[1],xys);
    
    cv::vconcat(xys,cv::Mat::ones(cv::Size(xys.cols,1),CV_32F),xys);
    cv::Mat uv=cv::Mat(M)*xys;
    uv-=1.;
    cv::Mat xi=uv.row(0);
    xi=xi.reshape(1,con_pts.rows);
    //xi=xi.reshape(1,1);
    cv::Mat yi=uv.row(1);
    yi=yi.reshape(1,con_pts.rows);
    //yi=yi.reshape(1,1);
     
    cv::Mat wimg;
    cv::remap(img,wimg,uv.row(0),uv.row(1),cv::INTER_LINEAR);
    //std::cout<<"wimg size "<<wimg.size()<<std::endl;
    return wimg;
}
cv::Mat ismarsd_images(const cv::Mat& dwdp,const cv::Mat& nIx,const cv::Mat& nIy,
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
cv::Mat ismarhessian_a(const cv::Mat& vIdwdp,const int np,const int w)
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
cv::Mat ismarsd_update(const cv::Mat& vIdwdp,const cv::Mat& err_img,const int np,
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
