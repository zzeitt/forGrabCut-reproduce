#include "grabcut_metrics.h"
#include <iostream>

using namespace std;

void calcMetrics(Mat img_1, Mat img_2, int x, int y, int w, int h) {
  int i_acc_count = 0;
  // int i_total = img_1.rows * img_1.cols;
  int i_total = w * h;
  int i_inter_count = 0;
  int i_union_count = 0;
  threshold(img_1, img_1, 254, 255, THRESH_BINARY);
  threshold(img_2, img_2, 254, 255, THRESH_BINARY);
  for (int r = y; r < y + h; r++) {
    for (int c = x; c < x + w; c++) {
      uchar u_1 = img_1.at<uchar>(r, c);
      uchar u_2 = img_2.at<uchar>(r, c);
      if (u_1 == u_2) i_acc_count++;
      if (u_1 == 255 && u_2 == 255) i_inter_count++;
      if (u_1 == 255 || u_2 == 255) i_union_count++;
    }
  }
  cout << endl;
  cout << " ** Accuracy **: " << (double)i_acc_count / i_total << endl;
  cout << " ** Jaccard **: " << (double)i_inter_count / i_union_count << endl;
  cin.get();
}
