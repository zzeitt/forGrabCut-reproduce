#include "grabcut_graph.h"

GrabCutGraph::GrabCutGraph(Mat img_src_arg, Mat mask_alpha_arg)
    : i_vertex_count{img_src_arg.rows * img_src_arg.cols},
      i_edge_count{2 * (4 * img_src_arg.cols * img_src_arg.rows -
                        3 * (img_src_arg.cols + img_src_arg.rows) + 2)},
      d_beta{0.0},
      graph_to_cut{i_vertex_count, i_edge_count} {
  // 拷贝原图和掩膜
  img_src_copy = img_src_arg;
  mask_alpha_copy = mask_alpha_arg;
  // 初始化各判罚矩阵
  mat_penalty_r_fgd = Mat::zeros(img_src_arg.size(), CV_64FC1);
  mat_penalty_r_bgd = Mat::zeros(img_src_arg.size(), CV_64FC1);
  mat_penalty_b_left = Mat::zeros(img_src_arg.size(), CV_64FC1);
  mat_penalty_b_up = Mat::zeros(img_src_arg.size(), CV_64FC1);
  mat_penalty_b_upleft = Mat::zeros(img_src_arg.size(), CV_64FC1);
  mat_penalty_b_upright = Mat::zeros(img_src_arg.size(), CV_64FC1);
  // 计算beta
  calcBeta();
}

void GrabCutGraph::calcBeta() {
  // 四个方向说明：
  //
  //   upleft   up  upright
  //          \ | /
  //   left —— NODE

  double d_square_sum = 0.0;
  for (int r = 0; r < img_src_copy.rows; r++) {
    for (int c = 0; c < img_src_copy.cols; c++) {
      Vec3b pix = img_src_copy.at<Vec3b>(r, c);
      if (c >= 1) {  // left
        Vec3b pix_diff = pix - img_src_copy.at<Vec3b>(r, c - 1);
        d_square_sum += pix_diff.dot(pix_diff);
      }
      if (r >= 1 && c >= 1) {  // upleft
        Vec3b pix_diff = pix - img_src_copy.at<Vec3b>(r - 1, c - 1);
        d_square_sum += pix_diff.dot(pix_diff);
      }
      if (r >= 1) {  // up
        Vec3b pix_diff = pix - img_src_copy.at<Vec3b>(r - 1, c);
        d_square_sum += pix_diff.dot(pix_diff);
      }
      if (r >= 1 && c <= img_src_copy.cols - 2) {  // upright
        Vec3b pix_diff = pix - img_src_copy.at<Vec3b>(r - 1, c + 1);
        d_square_sum += pix_diff.dot(pix_diff);
      }
    }
  }
  d_square_sum /= (double)i_edge_count;
  d_beta = (double)1 / (2 * d_square_sum);
}

void GrabCutGraph::calcWeight(GMM gmm_fgd_arg, GMM gmm_bgd_arg) {
  double d_gamma = 50.0;          // 计算边界判罚的一个比例参数
  double d_lambda = 9 * d_gamma;  // 确定性前/背景区域判罚值
  for (int r = 0; r < img_src_copy.rows; r++) {
    for (int c = 0; c < img_src_copy.cols; c++) {
      Vec3b pix = img_src_copy.at<Vec3b>(r, c);
      // 计算源汇权重
      if (mask_alpha_copy.at<uchar>(r, c) == GC_PR_BGD ||
          mask_alpha_copy.at<uchar>(r, c) == GC_PR_FGD) {
        mat_penalty_r_fgd.at<double>(r, c) =
            -log(gmm_bgd_arg.sumProbability(pix));
        // 属于前景的概率越高，折合所得分贝对数越小，故这里前/背景对调
        mat_penalty_r_bgd.at<double>(r, c) =
            -log(gmm_fgd_arg.sumProbability(pix));
      } else if (mask_alpha_copy.at<uchar>(r, c) == GC_BGD) {
        mat_penalty_r_fgd.at<double>(r, c) = 0;
        mat_penalty_r_bgd.at<double>(r, c) = d_lambda;
      } else if (mask_alpha_copy.at<uchar>(r, c) == GC_FGD) {
        mat_penalty_r_fgd.at<double>(r, c) = d_lambda;
        mat_penalty_r_bgd.at<double>(r, c) = 0;
      }
      //////////////////////////////////////////////////////
      // 计算邻边权重
      if (c >= 1) {  // left
        Vec3b pix_diff = pix - img_src_copy.at<Vec3b>(r, c - 1);
        mat_penalty_b_left.at<double>(r, c) =
            d_gamma * exp(-d_beta * pix_diff.dot(pix_diff));
      }
      if (r >= 1 && c >= 1) {  // upleft
        Vec3b pix_diff = pix - img_src_copy.at<Vec3b>(r - 1, c - 1);
        mat_penalty_b_upleft.at<double>(r, c) =
            d_gamma * exp(-d_beta * pix_diff.dot(pix_diff));
      }
      if (r >= 1) {  // up
        Vec3b pix_diff = pix - img_src_copy.at<Vec3b>(r - 1, c);
        mat_penalty_b_up.at<double>(r, c) =
            d_gamma * exp(-d_beta * pix_diff.dot(pix_diff));
      }
      if (r >= 1 && c <= img_src_copy.cols - 2) {  // upright
        Vec3b pix_diff = pix - img_src_copy.at<Vec3b>(r - 1, c + 1);
        mat_penalty_b_upright.at<double>(r, c) =
            d_gamma * exp(-d_beta * pix_diff.dot(pix_diff));
      }
    }
  }
}

void GrabCutGraph::assignWeight() {
  for (int r = 0; r < img_src_copy.rows; r++) {
    for (int c = 0; c < img_src_copy.cols; c++) {
      int i_vertex_id = graph_to_cut.add_node();
      // 分配源汇权重
      double d_weight_source = mat_penalty_r_fgd.at<double>(r, c);
      double d_weight_sink = mat_penalty_r_bgd.at<double>(r, c);
      graph_to_cut.add_tweights(i_vertex_id, d_weight_source, d_weight_sink);

      // 分配邻边权重
      double d_weight_edge;
      if (c >= 1) {  // left
        d_weight_edge = mat_penalty_b_left.at<double>(r, c);
        graph_to_cut.add_edge(i_vertex_id, i_vertex_id - 1, d_weight_edge,
                              d_weight_edge);
      }
      if (r >= 1 && c >= 1) {  // upleft
        d_weight_edge = mat_penalty_b_upleft.at<double>(r, c);
        graph_to_cut.add_edge(i_vertex_id, i_vertex_id - img_src_copy.cols - 1,
                              d_weight_edge, d_weight_edge);
      }
      if (r >= 1) {  // up
        d_weight_edge = mat_penalty_b_up.at<double>(r, c);
        graph_to_cut.add_edge(i_vertex_id, i_vertex_id - img_src_copy.cols,
                              d_weight_edge, d_weight_edge);
      }
      if (r >= 1 && c <= img_src_copy.cols - 2) {  // upright
        d_weight_edge = mat_penalty_b_upright.at<double>(r, c);
        graph_to_cut.add_edge(i_vertex_id, i_vertex_id - img_src_copy.cols + 1,
                              d_weight_edge, d_weight_edge);
      }
    }
  }
}

void GrabCutGraph::doMinimumCut() {
  // 通过最大流实现最小割
  graph_to_cut.maxflow();
  // 注意，要开始更新mask_alpha了
  for (int r = 0; r < img_src_copy.rows; r++) {
    for (int c = 0; c < img_src_copy.cols; c++) {
      int i_id_temp = r * img_src_copy.cols + c;
      if (mask_alpha_copy.at<uchar>(r, c) == GC_PR_BGD ||
          mask_alpha_copy.at<uchar>(r, c) == GC_PR_FGD) {
        if (graph_to_cut.what_segment(i_id_temp) ==
            Graph<double, double, double>::SOURCE) {
          mask_alpha_copy.at<uchar>(r, c) = GC_PR_FGD;
        } else {
          mask_alpha_copy.at<uchar>(r, c) = GC_PR_BGD;
        }
      }
    }
  }
}

Mat GrabCutGraph::getMaskAlpha() { return mask_alpha_copy; }
