#pragma once
#pragma warning(disable : 4996)
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "grabcut_method.h"

using namespace cv;
using namespace std;

class GrabCutClient {
 private:
  Mat img_src;              // 原图片
  Mat img_src_2;            // 用于用户操作的原图拷贝
  Rect rect_fgd;            // 前景矩形框
  Mat mask_alpha;           // 所属类别的掩膜
  Mat mask_fgd;             // 前景专属掩膜
  Mat img_dst;              // 生成图片
  bool b_opencv;            // 是否执行OpenCV内置算法
  int i_iterate;            // 迭代次数
  int i_comp;               // GMM中高斯组分个数
  const String win_src;     // 原图窗口
  const String win_dst;     // 生成图窗口
  GrabCutMethod gc_method;  // 算法执行类
 public:
  GrabCutClient(String file_path, bool b_opencv_arg, int i_comp_arg,
                int i_iterate_arg);
  void redrawSrcImage();  // 交互相关
  void onMouseMember(int event, int x, int y, int flags, void* userdata);
  static void onMouse(int event, int x, int y, int flags, void* userdata);
  void iterateLabelMask();  // 前景掩膜处理
  void showDstImage();      // 结果显示
  void saveTwoImages(string s_date, string s_time_elapse);  // 保存图片
};
