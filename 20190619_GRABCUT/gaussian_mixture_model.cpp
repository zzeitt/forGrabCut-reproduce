#include "gaussian_mixture_model.h"

Gau::Gau(int dim) : i_dim{dim} {
  mat_mean = Mat::zeros(1, i_dim, CV_64FC1);  // 行向量
  mat_cov = Mat::zeros(i_dim, i_dim, CV_64FC1);
  f_weight = 0.0;
}

void Gau::addPixel(Vec3b pix) { vec_pixs_gau.push_back(pix); }

void Gau::doComputation() {
  Mat mat_pixs_gau_channel(vec_pixs_gau);
  int i_rows_new = mat_pixs_gau_channel.rows * mat_pixs_gau_channel.cols;
  mat_pixs_gau = mat_pixs_gau_channel.reshape(1, i_rows_new);  // 通道变为1
  // 计算均值和协方差
  calcCovarMatrix(mat_pixs_gau, mat_cov, mat_mean, COVAR_ROWS | COVAR_NORMAL);
  mat_cov /= i_rows_new;
  //cout << "matrix:" << endl;
  //cout << mat_pixs_gau << endl;
  //cout << "cov:" << endl;
  //cout << mat_cov << endl;
  //cout << "mean:" << endl;
  //cout << mat_mean << endl;
}

GMM::GMM(int n, int d)
    : i_comp_count{n},
      i_comp_dim{d},
      vec_comp_gau{i_comp_count, Gau(i_comp_dim)} {}
