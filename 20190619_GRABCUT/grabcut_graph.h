#pragma once
#include <opencv2/opencv.hpp>
#include "VladimirKolmogorov/graph.h"
#include "gaussian_mixture_model.h"

using namespace cv;

class GrabCutGraph {
 private:
  int i_vertex_count;                          // 图的定点个数
  int i_edge_count;                            // 图的边个数
  double d_beta;                               // 邻边判罚的beta参数
  double d_gamma;                              // 邻边判罚的gamma参数
  Mat img_src_copy;                            // 原图 - 副本
  Mat mask_alpha_copy;                         // 掩膜 - 副本
  Graph<double, double, double> graph_to_cut;  // 待割的图，三个double
                                               // 为容量和流量的数据类型
  double d_energy;  // 图割后花费的能量
 public:
  GrabCutGraph(Mat img_src_arg, Mat mask_alpha_arg);
  void initBetaAndNode();                          // 计算beta
  double calcBoundrayPenalty(Vec3b pix_diff_arg);  //计算邻边权重
  void assignBoundaryWeight();                     //分配邻边权重
  void assignRegionalWeight(GMM gmm_fgd_arg,
                            GMM gmm_bgd_arg);  // 分配区域的权重
  void doMinimumCut();                         // 进行最小割
  Mat getMaskAlpha();  // 访问最小割之后的索引矩阵
};
