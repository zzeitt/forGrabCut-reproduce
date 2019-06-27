#include "gaussian_mixture_model.h"

Gau::Gau(int dim) : i_dim{dim}, i_pix_count{0} {
  mat_mean = Mat::zeros(1, i_dim, CV_64FC1);  // 行向量
  mat_cov = Mat::zeros(i_dim, i_dim, CV_64FC1);
  f_weight = 0.0;
}

void Gau::addPixel(Vec3b pix) {
  vec_pixs_gau.push_back(pix);
  i_pix_count = vec_pixs_gau.size();  // 更新像素个数值
}

void Gau::doComputation(int i_pix_gmm_count) {
  // 权重赋值
  f_weight = (float)i_pix_count / i_pix_gmm_count;
  // 向量转矩阵
  Mat mat_pixs_gau_channel(vec_pixs_gau);
  int i_rows_new = mat_pixs_gau_channel.rows * mat_pixs_gau_channel.cols;
  mat_pixs_gau = mat_pixs_gau_channel.reshape(1, i_rows_new);  // 通道变为1
  // 计算均值和协方差
  calcCovarMatrix(mat_pixs_gau, mat_cov, mat_mean, COVAR_ROWS | COVAR_NORMAL);
  mat_cov /= i_rows_new;
  cout << "==================Gau==================" << endl;
  cout << "cov:" << endl;
  cout << mat_cov << endl;
  cout << "mean:" << endl;
  cout << mat_mean << endl;
  cout << "weight:" << endl;
  cout << f_weight << endl;
}

GMM::GMM(int n, int d) : i_comp_count{n}, i_comp_dim{d}, i_pix_gmm_count{0} {
  for (int i = 0; i < i_comp_count; i++) {
    Gau gau_iter(i_comp_dim);
    vec_comp_gau.push_back(gau_iter);
  }
}

void GMM::fitGMM(Mat mat_k, vector<Vec3b> vec_pixs) {
  // mat_k是n行x1列的索引矩阵
  // 先将像素信息传给GMM
  i_pix_gmm_count = mat_k.rows;
  for (int i = 0; i < i_pix_gmm_count; i++) {
    int i_comp_index = mat_k.at<int>(i, 0);
    Vec3b pix_to_add = vec_pixs[i];
    vec_comp_gau[i_comp_index].addPixel(pix_to_add);
  }
  // GMM的每个高斯函数组分各自学习
  for (int i = 0; i < i_comp_count; i++) {
    vec_comp_gau[i].doComputation(i_pix_gmm_count);
  }
}
