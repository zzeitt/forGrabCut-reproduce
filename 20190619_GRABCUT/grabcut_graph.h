#pragma once
#include <opencv2/opencv.hpp>
#include "VladimirKolmogorov/graph.h"
#include "gaussian_mixture_model.h"

using namespace cv;

class GrabCutGraph {
 private:
  int i_vertex_count;                          // 图的定点个数
  int i_edge_count;                            // 图的边个数
  double d_beta;                               // 边界判罚的beta参数
  Mat img_src_copy;                            // 原图 - 副本
  Mat mask_alpha_copy;                         // 掩膜 - 副本
  Mat mat_penalty_r_fgd;                       // 区域判罚：前景
  Mat mat_penalty_r_bgd;                       // 区域判罚：背景
  Mat mat_penalty_b_left;                      // 边界判罚：左
  Mat mat_penalty_b_up;                        // 边界判罚：上
  Mat mat_penalty_b_upleft;                    // 边界判罚：左上
  Mat mat_penalty_b_upright;                   // 边界判罚：右上
  Graph<double, double, double> graph_to_cut;  // 待割的图，三个double
                                               // 为容量和流量的数据类型
  double d_energy;  // 图割后花费的能量
 public:
  GrabCutGraph(Mat img_src_arg, Mat mask_alpha_arg);
  void calcBeta();                                    // 计算beta
  void calcWeight(GMM gmm_fgd_arg, GMM gmm_bgd_arg);  // 建立边的权重
  void assignWeight();                                // 分配权重到图上
  void doMinimumCut();                                // 进行最小割
  Mat getMaskAlpha();  // 访问最小割之后的索引矩阵
};
