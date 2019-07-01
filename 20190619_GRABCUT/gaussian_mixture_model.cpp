#include "gaussian_mixture_model.h"

/*******************高斯函数*******************/
Gau::Gau(int dim) : i_dim{dim}, i_pix_count{0} {
  mat_mean = Mat::zeros(1, i_dim, CV_64FC1);  // 行向量
  mat_cov = Mat::zeros(i_dim, i_dim, CV_64FC1);
  d_weight = 0.0;
}

void Gau::addPixel(Vec3b pix) {
  vec_pixs_gau.push_back(pix);
  i_pix_count = vec_pixs_gau.size();  // 更新像素个数值
}

void Gau::clearPixels() {
  vec_pixs_gau.clear();
  i_pix_count = vec_pixs_gau.size();
}

void Gau::doComputation(int i_pix_gmm_count) {
  // 权重赋值
  d_weight = (double)i_pix_count / (double)i_pix_gmm_count;
  // 向量转矩阵
  try {
    Mat mat_pixs_gau_channel(vec_pixs_gau);
    if (vec_pixs_gau.empty()) throw "This Gau is empty!";
    int i_rows_new = mat_pixs_gau_channel.rows * mat_pixs_gau_channel.cols;
    mat_pixs_gau = mat_pixs_gau_channel.reshape(1, i_rows_new);  // 通道变为1
    // 计算均值和协方差
    calcCovarMatrix(mat_pixs_gau, mat_cov, mat_mean, COVAR_ROWS | COVAR_NORMAL);
    mat_cov /= i_rows_new;
  } catch (char* str) {
    mat_mean = Mat::zeros(1, i_dim, CV_64FC1);  // 行向量
    mat_cov = Mat::ones(i_dim, i_dim, CV_64FC1) * 0.01;
  } catch (...) {
    mat_mean = Mat::zeros(1, i_dim, CV_64FC1);  // 行向量
    mat_cov = Mat::ones(i_dim, i_dim, CV_64FC1) * 0.01;
  }
  // 输出测试
  // cout << "==================Gau==================" << endl;
  // cout << "【cov】:" << endl;
  // cout << mat_cov << endl;
  // cout << "【mean】:" << endl;
  // cout << mat_mean << endl;
  // cout << "【weight】:" << endl;
  // cout << d_weight << endl;
}

double Gau::getWeight() { return d_weight; }

double Gau::calcProbability(Vec3b pix) {
  // 像素向量转矩阵
  Mat mat_pix(1, 3, CV_64FC1);
  for (int i = 0; i < 3; i++) {
    mat_pix.at<double>(0, i) = pix[i];
  }
  // 计算一些中间量
  double det_cov = determinant(mat_cov);
  double d_mul_1 =
      1.0 / (pow(2 * CV_PI, (double)i_dim / 2) * (determinant(mat_cov)));
  Mat mat_x_mu = mat_pix - mat_mean;
  Mat mat_cov_inv = mat_cov.inv();
  Mat mat_exp = mat_x_mu * mat_cov_inv * mat_x_mu.t();
  double d_exp = determinant(mat_exp);
  double d_mul_2 = exp(-0.5 * d_exp);
  double d_ret = d_mul_1 * d_mul_2;
  return d_ret;
}

/*******************高斯混合模型*******************/
GMM::GMM(int n, int d) : i_comp_count{n}, i_comp_dim{d}, i_pix_gmm_count{0} {
  for (int i = 0; i < i_comp_count; i++) {
    Gau gau_iter(i_comp_dim);
    vec_comp_gau.push_back(gau_iter);
  }
}

void GMM::fitGMM(Mat mat_k, vector<Vec3b> vec_pixs) {
  // mat_k是n行x1列的索引矩阵

  i_pix_gmm_count = mat_k.rows;
  for (int i = 0; i < i_comp_count; i++) {
    vec_comp_gau[i].clearPixels();
  }
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

int GMM::findMostLikelyGau(Vec3b pix) {
  int i_comp_max_index = 0;
  double d_prob_max = 0.0;
  for (int i = 0; i < i_comp_count; i++) {
    double d_prob_temp = vec_comp_gau[i].calcProbability(pix);
    if (d_prob_temp > d_prob_max) {
      d_prob_max = d_prob_temp;
      i_comp_max_index = i;
    }
  }
  return i_comp_max_index;
}

double GMM::sumProbability(Vec3b pix) {
  double d_sum = 0.0;
  for (int i = 0; i < i_comp_count; i++) {
    d_sum += vec_comp_gau[i].getWeight() * vec_comp_gau[i].calcProbability(pix);
  }
  return d_sum;
}
