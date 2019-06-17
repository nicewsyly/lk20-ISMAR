#include "utils.hpp"

void get_videos( const std::string& path, const std::string& exd, std::vector<std::string>& files )
{
    DIR* dir=opendir(path.c_str());
    dirent* p=nullptr;
    while((p=readdir(dir))!=nullptr)
    {
        P("name ",std::string(p->d_name+strlen(p->d_name)-3));
        if(p->d_name[0]=='.')
            continue;
        if(p->d_type!=DT_DIR&&strcmp(p->d_name+strlen(p->d_name)-3,"mov")==0)
        {
            files.push_back(path+'/'+std::string(p->d_name));
        }
        if(p->d_type==DT_DIR)
        {
            get_videos(path+'/'+p->d_name,exd,files);
        }
    }
    closedir(dir);
}
struct mparam
{
    cv::Mat img;
    std::vector<cv::Point2f> contour;
};
void on_mouse(int event,int x,int y,int flags,void* param)
{
    mparam* res=(mparam*)param;
    if(event==cv::EVENT_LBUTTONDOWN)
    {
        res->contour.push_back(cv::Point2f(x,y));
        for(unsigned int ri=1;ri<res->contour.size();++ri)
        {
            cv::line(res->img,res->contour[ri-1],res->contour[ri],cv::Scalar(0,0,255),5);
        }
         
    }
}
void get_contour(const cv::Mat& img,std::vector<cv::Point2f>& temp_contour)
{
    mparam res({img,temp_contour});
    cv::namedWindow("image");
    cv::setMouseCallback("image",on_mouse,(void*)(&res));
    while(1)
    {
        cv::imshow("image",img);
        if(cv::waitKey(1)=='q')
        {
            break;
        }
    }
    cv::destroyAllWindows();
    std::swap(temp_contour,res.contour);
    P("temp_contour",temp_contour.size());
}
cv::Mat DrawInlier(const cv::Mat &src1, const cv::Mat &src2, std::vector<cv::KeyPoint> &kpt1, std::vector<cv::KeyPoint> &kpt2, std::vector<cv::DMatch> &inlier, int type) 
{
        const int height = std::max(src1.rows, src2.rows);
        const int width = src1.cols + src2.cols;
        cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
        src1.copyTo(output(cv::Rect(0, 0, src1.cols, src1.rows)));
        src2.copyTo(output(cv::Rect(src1.cols, 0, src2.cols, src2.rows)));

        if (type == 1)
        {
                for (size_t i = 0; i < inlier.size(); i++)
                {
                        cv::Point2f left = kpt1[inlier[i].queryIdx].pt;
                        cv::Point2f right = (kpt2[inlier[i].trainIdx].pt + cv::Point2f((float)src1.cols, 0.f));
                        cv::line(output, left, right, cv::Scalar(0, 255, 255));
                }
        }
        else if (type == 2)
        {
                for (size_t i = 0; i < inlier.size(); i++)
                {
                        cv::Point2f left = kpt1[inlier[i].queryIdx].pt;
                        cv::Point2f right = (kpt2[inlier[i].trainIdx].pt + cv::Point2f((float)src1.cols, 0.f));
                        cv::line(output, left, right, cv::Scalar(255, 0, 0));
                }

                for (size_t i = 0; i < inlier.size(); i++)
                {
                        cv::Point2f left = kpt1[inlier[i].queryIdx].pt;
                        cv::Point2f right = (kpt2[inlier[i].trainIdx].pt + cv::Point2f((float)src1.cols, 0.f));
                        cv::circle(output, left, 1, cv::Scalar(0, 255, 255), 2);
                        cv::circle(output, right, 1, cv::Scalar(0, 255, 0), 2);
                }
        }

        return output;
}

