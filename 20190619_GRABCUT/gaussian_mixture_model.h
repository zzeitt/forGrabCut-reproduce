#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

/*******************高斯函数*******************/
class Gau {
 private:
  int i_dim;                   // 维数
  Mat mat_mean;                // 均值矩阵：1行xn列
  Mat mat_cov;                 // 协方差矩阵
  double d_weight;             // 权重
  vector<Vec3b> vec_pixs_gau;  // 存储每个高斯函数对应的像素们
  Mat mat_pixs_gau;            // 存储为矩阵形式：n行x3列
  int i_pix_count;             // 每个函数的像素个数
 public:
  Gau(int dim);
  void addPixel(Vec3b pix);                 // 为高斯堆添加像素
  void doComputation(int i_pix_gmm_count);  // 根据添加的像素计算13个参数
  double calcProbability(Vec3b pix);  // 输入单个像素，输出对应概率
};

/*******************高斯混合模型*******************/
class GMM {
 private:
  int i_comp_count;          // 组分个数
  int i_comp_dim;            // 组分维数
  vector<Gau> vec_comp_gau;  // 组分向量
  int i_pix_gmm_count;       // 模型拟合的总像素个数
 public:
  GMM(int n, int d);
  void fitGMM(Mat mat_k, vector<Vec3b> vec_pixs);  // 根据索引和像素向量拟合模型
  int findMostLikelyGau(Vec3b pix);  // 输入单个像素，找到概率最大的高斯堆
  double sumProbability(Vec3b pix);  // 累加单个像素在各高斯函数的概率值
};
