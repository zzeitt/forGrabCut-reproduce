#include "grabcut_client.h"
#include "grabcut_metrics.h"

int main() {
  GrabCutClient gc_client("images/white-bird-2-test.jpg", false /*b_opencv*/,
                          1 /*i_comp*/, 1 /*i_iterate*/);

  // ŒÛ≤Ó∂»¡ø
  // Mat img_mine =
  //    imread("results/2019-07-02-05-32-28_24_s_Mine_5_iter_1_comp_alpha.jpg",
  //           IMREAD_GRAYSCALE);
  // Mat img_gt = imread("images/white-bird-2-ground-truth.jpg",
  // IMREAD_GRAYSCALE); calcMetrics(img_mine, img_gt, 161, 36, 163, 210);
  return 0;
}
