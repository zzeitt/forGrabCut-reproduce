#include "grabcut_client.h"

int main() {
  // GrabCutClient gc_client(
  //    "D:\\NiseEngFolder\\MyDocument\\MyPictures\\forMiscellaneous\\20190619_"
  //    "CV_grabcut\\owl.jpg");
  Mat img_temp(2, 3, CV_8UC3);
  img_temp.at<Vec3b>(0, 0) = Vec3b(0, 255, 0);
  img_temp.at<Vec3b>(0, 1) = Vec3b(0, 0, 255);
  img_temp.at<Vec3b>(0, 2) = Vec3b(0, 255, 1);
  img_temp.at<Vec3b>(1, 0) = Vec3b(0, 127, 0);
  img_temp.at<Vec3b>(1, 1) = Vec3b(0, 0, 127);
  img_temp.at<Vec3b>(1, 2) = Vec3b(0, 127, 1);
  Mat mask_alpha(img_temp.size(), CV_8UC1, Scalar(GC_FGD));
  mask_alpha.at<uchar>(1, 0) = GC_BGD;
  mask_alpha.at<uchar>(1, 1) = GC_BGD;
  mask_alpha.at<uchar>(1, 2) = GC_BGD;

  GrabCutMethod gc_method(2, img_temp.dims, 2, img_temp.dims);
  gc_method.initPixelsVec(img_temp, mask_alpha);
  gc_method.clusterPixels();

  return 0;
}
