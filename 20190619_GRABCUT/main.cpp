#include "grabcut_client.h"

int main() {
  //GrabCutClient gc_client(
  //    "D:\\NiseEngFolder\\MyDocument\\MyPictures\\forMiscellaneous\\20190619_"
  //    "CV_grabcut\\owl.jpg");
  Mat img_temp(1, 2, CV_8UC3);
  img_temp.at<Vec3b>(0, 0) = Vec3b(0, 255, 0);
  img_temp.at<Vec3b>(0, 1) = Vec3b(0, 0, 255);
  Mat mask_alpha(img_temp.size(), CV_8UC1, Scalar(GC_FGD));
  GrabCutMethod gc_method(2, img_temp.dims, 2, img_temp.dims);
  gc_method.initPixelsVec(img_temp, mask_alpha);
  gc_method.clusterPixels();

  return 0;
}