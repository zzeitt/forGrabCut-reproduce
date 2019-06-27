#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "gaussian_mixture_model.h"

using namespace cv;
using namespace std;

class GrabCutMethod {
 private:
  int i_fgd_comp;             // 前景高斯组分个数
  int i_bgd_comp;             // 背景组分个数
  Mat img_src;                // 原图
  Mat mask_alpha;             // 标签矩阵
  vector<Vec3b> vec_pixs_fgd; // 前景像素向量
  vector<Vec3b> vec_pixs_bgd; // 背景向量
  Mat mat_pixs_fgd;           // 前景像素矩阵
  Mat mat_pixs_bgd;           // 背景矩阵
  Mat mat_pixs_fgd_k;         // 前景像素模型索引
  Mat mat_pixs_bgd_k;         // 背景索引
  GMM gmm_fgd;                // 前景高斯模型
  GMM gmm_bgd;                // 背景模型

 public:
  GrabCutMethod(int i_fgd_comp_arg, int i_fgd_comp_dim_arg, int i_bgd_comp_arg,
                int i_bgd_comp_dim_arg);
  void initPixelsVec(Mat img, Mat mask);
  void clusterPixels();
  void fitGMMs();
};
