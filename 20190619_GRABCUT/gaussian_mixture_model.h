#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/*******************高斯函数*******************/
class Gau {
private:
  int i_dim;                  // 维数
  Mat mat_mean;               // 均值矩阵
  Mat mat_cov;                // 协方差矩阵
  float f_weight;             // 权重
  vector<Vec3b> vec_pixs_gau; // 存储每个高斯函数对应的像素们
  Mat mat_pixs_gau;           // 存储为矩阵形式：n行x3列
  int i_pix_count;            // 每个函数的像素个数
public:
  Gau(int dim);
  void addPixel(Vec3b pix);
  void doComputation(int i_pix_gmm_count);
};

/*******************高斯混合模型*******************/
class GMM {
private:
  int i_comp_count;         // 组分个数
  int i_comp_dim;           // 组分维数
  vector<Gau> vec_comp_gau; // 组分向量
  int i_pix_gmm_count;          // 模型拟合的总像素个数
public:
  GMM(int n, int d);
  void fitGMM(Mat mat_k, vector<Vec3b> vec_pixs);
};
