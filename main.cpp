
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include<gdal_priv.h>;
#include<iostream>;
using namespace cv;
using namespace std;

#define NULL_VAL -32767

cv::Point2f getCorresLoc(const size_t r, const size_t c, const double* const  GT1, const double* const GT2){
	double X = GT1[0] + r * GT1[1] + c * GT1[2];
	double Y = GT1[3] + r * GT1[4] + c * GT1[5];

	double a1 = GT2[1], b1 = GT2[2], c1 = X - GT2[0];
	double a2 = GT2[4], b2 = GT2[5], c2 = Y - GT2[3];
	a1*=1e6; b1*=1e6; c1*=1e6;
	a2*=1e6; b2*=1e6; c2*=1e6;
	double det = b1*a2 - b2*a1;
	if(abs(det) < 1e-8){
		// cout << det << "  ";
		return cv::Point2f(-1,-1);
	}
	double xr = b1*c2 - b2*c1;
	double yc = c1*a2 - c2*a1;
	return cv::Point2f(xr/det, yc/det);
}

inline bool isInImage(const cv::Mat& img, const int r, const int c){
	return !(r < 0 || c<0 || r>= img.rows || c>= img.cols);
}

void showHist(const cv::Mat& img, const cv::Mat& mask){
	 /// 设定bin数目
 int histSize = 255;

 // get attitude range
   	double minv = 0.0, maxv = 0.0;
	double* minp = &minv;
	double* maxp = &maxv;
	minMaxIdx(img,minp,maxp, NULL, NULL, mask);
	cout << "min:" << minv << "  max:" << maxv << endl;

 float range[] = { minv, maxv } ;
 const float* histRange = { range };

 Mat hist;

 /// 计算直方图:
 calcHist( &img, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false );

 // 创建直方图画布
 int hist_w = 400; int hist_h = 400;
 int bin_w = cvRound( (double) hist_w/histSize );

 Mat histImage( hist_w, hist_h, CV_8UC3, Scalar( 0,0,0) );

 /// 将直方图归一化到范围 [ 0, histImage.rows ]
 normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

 /// 在直方图画布上画出直方图
 for( int i = 1; i < histSize; i++ )
   {
     line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                      Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                      Scalar( 0, 0, 255), 2, 8, 0  );
    }

 /// 显示直方图
 imshow("calcHist Demo", histImage );

 waitKey(0);
}

void getAttitude(const cv::Mat& img1, const cv::Mat& mask1, const double* const trans1, 
	const cv::Mat& img2, const cv::Mat& mask2, const double* const trans2, cv::Mat& diff, cv::Mat& mask){
	diff = cv::Mat (img1.size(), CV_32F, cv::Scalar(NULL_VAL));
	mask = cv::Mat(img1.size(), CV_8U, Scalar(0));

	int count = 0, f1=0, f2=0, f3=0;
	for(size_t r=0; r<diff.rows; ++r){
		for(size_t c=0; c<diff.cols; ++c){
			if(mask1.at<uchar>(r,c) < 1) { f1++;continue;}
			float height1 = img1.at<float>(r,c);
			cv::Point2f rc2 = getCorresLoc(r, c, trans1, trans2);
			// cout << "(" << rc2.x << "," << rc2.y << ") ";
			if(!isInImage(img2, rc2.x, rc2.y)) {f2++; continue;}
			if(mask2.at<uchar>(rc2.x, rc2.y) < 1) {f3++; continue;}
			float height2 = img2.at<float>(rc2.x, rc2.y);
			diff.at<float>(r,c) = height2 - height1;
			mask.at<uchar>(r,c) = 255;
			count++;
		}
	}
	cout << "diff : " << count << "  f1:" << f1 << "   f2:" << f2 << "   f3:" << f3 << endl;
}

void showDEMImage(const cv::Mat& img, const cv::Mat& mask){
	Mat result(img.size(),CV_32F);
	double scale = 1080.0/max(img.cols, img.rows);
   	Size dsize = Size(img.cols*scale, img.rows*scale);
   	normalize(img,result,1,0,cv::NORM_MINMAX, -1, mask);
	Mat imagedst = Mat(dsize, CV_32F);
	cv::resize(result, imagedst, dsize);
	namedWindow("result", cv::WINDOW_FULLSCREEN);
    imshow("result",imagedst);
	waitKey(0);
}

void showResizeImage(const cv::Mat& img){
	double scale = 1080.0/max(img.cols, img.rows);
   	Size dsize = Size(img.cols*scale, img.rows*scale);
	Mat imagedst = Mat(dsize, CV_32F);
	cv::resize(img, imagedst, dsize);
	namedWindow("result", cv::WINDOW_FULLSCREEN);
    imshow("result",imagedst);
	waitKey(0);
}

bool readTifData(string filePath, cv::Mat& img, cv::Mat& mask, double* trans){
	GDALAllRegister();
	GDALDataset *poDataset = (GDALDataset*)GDALOpen(filePath.c_str(),GA_ReadOnly);

	// trans[0] /* top left x */
	// trans[1] /* w-e pixel resolution */
	// trans[2] /* 0 */
	// trans[3] /* top left y */
	// trans[4] /* 0 */
	// trans[5] /* n-s pixel resolution (negative value) */
	if(poDataset->GetGeoTransform(trans) != CE_None){
		cout << "get geo trans fail" << endl;
		return false;
	}

	//获取影像的宽高，波段数
    int width = poDataset->GetRasterXSize();
    int height = poDataset->GetRasterYSize();
    int nband = poDataset->GetRasterCount();

    GDALRasterBand *bandData;
    float *p = new float[width*height];
    vector<Mat> imagesT;

    //遍历波段，读取到mat向量容器里.注意顺序
    for(int i = 1;i <= nband;i++)
    {
        bandData = poDataset->GetRasterBand(i);
        GDALDataType DataType = bandData->GetRasterDataType();
        bandData->RasterIO(GF_Read,0,0,width,height,p,width,height,DataType ,0,0);

		int CVDataType = nband == 1 ? CV_32F : CV_8U;
        Mat HT(height,width,CVDataType,p);
        imagesT.push_back(HT.clone());
    }

	if(nband == 1){
		// mask
		img = imagesT[0].clone();
		img.convertTo(mask,CV_8U);
   		cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);

		// get attitude range
   		double minv = 0.0, maxv = 0.0;
		double* minp = &minv;
		double* maxp = &maxv;
		minMaxIdx(img,minp,maxp, NULL, NULL, mask);
		cout << "min:" << minv << "  max:" << maxv << endl;
		// showHist(img, mask);
		// showDEMImage(img, mask);
	}else{
		//多通道融合
    	merge(imagesT,img);
		img.convertTo(mask,CV_8U);
		cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);
	}
	
	cout << "read file suc." << filePath << endl;
	return true;
}

inline void showTrans(double* d, string name){
	cout << name << ": " << d[0] << "," << d[1] << "," << d[2] << "," << d[3] << "," << d[4] << "," << d[5] << endl;
}

int main()
{
	const string filePathDEM1 = "/media/ll/DATA/SLAM_Data/DEM_Data/DEM1.tif";
	const string filePathRGB1 = "/media/ll/DATA/SLAM_Data/DEM_Data/ZHENGSHE1.tif";
	const string filePathDEM2 = "/media/ll/DATA/SLAM_Data/DEM_Data/DEM2.tif";
	const string filePathRGB2 = "/media/ll/DATA/SLAM_Data/DEM_Data/ZHENGSHE2.tif";

	cv::Mat img_DEM1, img_RGB1, mask_DEM1, mask_RGE1;
	cv::Mat img_DEM2, img_RGB2, mask_DEM2, mask_RGB2;
	double trans_DEM1[6],  trans_RGB1[6], trans_DEM2[6], trans_RGB2[6];

	bool isSuc_DEM1 = readTifData(filePathDEM1, img_DEM1, mask_DEM1, trans_DEM1);
	bool isSuc_RGB1 = readTifData(filePathRGB1, img_RGB1, mask_RGE1, trans_RGB1);
	bool isSuc_DEM2 = readTifData(filePathDEM2, img_DEM2, mask_DEM2, trans_DEM2);
	bool isSuc_RGB2 = readTifData(filePathRGB2, img_RGB2, mask_RGB2, trans_RGB2);

	showTrans(trans_DEM1, "trans_DEM1");
	showTrans(trans_RGB1, "trans_RGB1");
	showTrans(trans_DEM2, "trans_DEM2");
	showTrans(trans_RGB2, "trans_RGB2");

	showResizeImage(img_DEM1);
	showResizeImage(img_RGB1);
	showResizeImage(img_DEM2);
	showResizeImage(img_RGB2);
	showResizeImage(mask_DEM1);
	showResizeImage(mask_RGE1);
	showResizeImage(mask_DEM2);
	showResizeImage(mask_RGB2);

	// cv::Mat roi(100, 100, CV_32F, Scalar(100.0));
	// roi.copyTo(img2.rowRange(img2.rows/2, img2.rows/2 + roi.rows).colRange(img2.cols/2, img2.cols/2 + roi.cols));
	// showDEMImage(img1, mask1);
	// showDEMImage(img2, mask2);

	if(!isSuc_DEM1 || !isSuc_RGB1 || !isSuc_DEM2 || !isSuc_RGB2){
		return -1;
	}

	cv::Mat diff, mask;
	getAttitude(img_DEM1, mask_DEM1, trans_DEM1, img_DEM2, mask_DEM2, trans_DEM2, diff, mask);
	showResizeImage(diff);
	showHist(diff, mask);
	showDEMImage(diff, mask);
	return 0;
}