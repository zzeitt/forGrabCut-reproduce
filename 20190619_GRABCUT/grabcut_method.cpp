#include "grabcut_method.h"

GrabCutMethod::GrabCutMethod(int i_fgd_comp_arg, int i_fgd_comp_dim_arg,
                             int i_bgd_comp_arg, int i_bgd_comp_dim_arg)
    : i_fgd_comp{i_fgd_comp_arg},
      i_bgd_comp{i_bgd_comp_arg},
      gmm_fgd{i_fgd_comp, i_fgd_comp_dim_arg},
      gmm_bgd{i_bgd_comp, i_bgd_comp_dim_arg} {}

void GrabCutMethod::initPixelsVec(Mat img, Mat mask) {
  img.copyTo(img_src);
  mask.copyTo(mask_alpha);
  for (int r = 0; r < img_src.rows; r++) {
    for (int c = 0; c < img_src.cols; c++) {
      if (mask_alpha.at<uchar>(r, c) == GC_BGD) {
        // 初始化背景像素组
        vec_pixs_bgd.push_back(img_src.at<Vec3b>(r, c));
      } else {
        // 初始化前景像素组
        vec_pixs_fgd.push_back(img_src.at<Vec3b>(r, c));
      }
    }
  }
}

void GrabCutMethod::clusterPixels() {
  // 向量转化为kmeans可接受的矩阵形式
  Mat mat_pixs_fgd_channel(vec_pixs_fgd);
  int i_rows_new_fgd = mat_pixs_fgd_channel.rows * mat_pixs_fgd_channel.cols;
  mat_pixs_fgd = mat_pixs_fgd_channel.reshape(1, i_rows_new_fgd);
  mat_pixs_fgd.convertTo(mat_pixs_fgd, CV_32F);
  Mat mat_pixs_bgd_channel(vec_pixs_bgd);
  int i_rows_new_bgd = mat_pixs_bgd_channel.rows * mat_pixs_bgd_channel.cols;
  mat_pixs_bgd = mat_pixs_bgd_channel.reshape(1, i_rows_new_bgd);
  mat_pixs_bgd.convertTo(mat_pixs_bgd, CV_32F);
  // KMeans聚类
  try {
    if (mat_pixs_fgd.empty() || mat_pixs_bgd.empty())
      throw "No elements to cluster!";
    kmeans(mat_pixs_fgd, i_fgd_comp, mat_pixs_fgd_k,
           TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10, 1.0),
           10, KMEANS_RANDOM_CENTERS);
    kmeans(mat_pixs_bgd, i_bgd_comp, mat_pixs_bgd_k,
           TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10, 1.0),
           10, KMEANS_RANDOM_CENTERS);
  } catch (char* str) {
    cout << "【Error】: " << str << endl;
  } catch (...) {
    cout << "【Error】: "
         << "KMeans failed..." << endl;
  }
}

void GrabCutMethod::fitTwoGMMs() {
  // GMM根据掩膜和像素进行学习拟合
  // cout << "==================FGD==================" << endl;
  gmm_fgd.fitGMM(mat_pixs_fgd_k, vec_pixs_fgd);
  // cout << "==================BGD==================" << endl;
  gmm_bgd.fitGMM(mat_pixs_bgd_k, vec_pixs_bgd);
}

void GrabCutMethod::updateTwoIndexMat() {
  for (int i = 0; i < mat_pixs_fgd_k.rows; i++) {
    Vec3b pix_iter = vec_pixs_fgd[i];
    mat_pixs_fgd_k.at<int>(i, 0) = gmm_fgd.findMostLikelyGau(pix_iter);
  }
  for (int i = 0; i < mat_pixs_bgd_k.rows; i++) {
    Vec3b pix_iter = vec_pixs_bgd[i];
    mat_pixs_bgd_k.at<int>(i, 0) = gmm_bgd.findMostLikelyGau(pix_iter);
  }
}

GrabCutGraph GrabCutMethod::initGraph() {
  return GrabCutGraph(img_src, mask_alpha);
}

void GrabCutMethod::updateMaskAlpha(GrabCutGraph& gc_graph_arg) {
  gc_graph_arg.assignRegionalWeight(gmm_fgd,
                                    gmm_bgd);  // 分配区域的权重
  gc_graph_arg.doMinimumCut();                 // 进行最小割
  mask_alpha = gc_graph_arg.getMaskAlpha();
}

Mat GrabCutMethod::getMaskAlpha() { return mask_alpha; }
