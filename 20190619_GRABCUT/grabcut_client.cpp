#include "grabcut_client.h"
#include <time.h>

using namespace std::chrono;

GrabCutClient::GrabCutClient(String file_path, int i_iterate_arg,
                             bool b_opencv_arg)
    : b_opencv{b_opencv_arg},
      win_src{"Image SRC"},
      win_dst{"Image DST"},
      gc_method{5, 3, 5, 3},
      i_iterate{i_iterate_arg} {
  /*******************初始化*******************/
  try {
    img_src = imread(file_path);
    if (img_src.empty()) {
      throw "Empty image!";
    }
  } catch (char* str) {
    cout << "【Error】: " << str << endl;
    Mat img_exception(200, 300, CV_8UC3, Scalar(127, 127, 127));
    img_src = img_exception;
  } catch (...) {
    cout << "【Error】: Image load failed!" << endl;
    Mat img_exception(200, 300, CV_8UC3, Scalar(127, 127, 127));
    img_src = img_exception;
  }
  mask_alpha = Mat::zeros(img_src.size(), CV_8UC1);  // 初始化为全GC_BGD=0
  Mat img_dst_temp(img_src.size(), CV_8UC3, Scalar(96, 96, 96));
  img_dst = img_dst_temp;  // 初始化结果图
  /*******************用户框选*******************/
  namedWindow(win_src, WINDOW_AUTOSIZE);
  imshow(win_src, img_src);
  setMouseCallback(win_src, onMouse, this);
  /*******************按键响应*******************/
  char c = waitKey(0);
  if (c == 'g') {
    auto start = high_resolution_clock::now();  // 计时开始
    if (b_opencv) {
      Mat mat_temp_1, mat_temp_2;
      grabCut(img_src, mask_alpha, rect_fgd, mat_temp_1, mat_temp_2, i_iterate,
              GC_INIT_WITH_RECT);
    } else {
      iterateLabelMask();
    }
    auto stop = high_resolution_clock::now();  //计时结束
    auto duration = duration_cast<seconds>(stop - start);
    cout << endl
         << " ===================>>>【共计用时" << duration.count() << "秒】"
         << endl;
    string s_time_elapse = to_string(duration.count());
    showDstImage();
    cout << endl << " ===================>>>【按 s 键保存】" << endl;
    char s = waitKey(0);
    if (s == 's') {
      // 生成时间戳
      auto now = time(nullptr);
      ostringstream os;
      os << put_time(gmtime(&now), "%F-%H-%M-%S");
      // 保存图片
      saveTwoImages(os.str(), s_time_elapse);
    }
    waitKey(0);
    destroyAllWindows();
  }
}

void GrabCutClient::redrawSrcImage() {
  img_src.copyTo(img_src_2);
  rectangle(img_src_2, rect_fgd, Scalar(0, 0, 255), 2);
  imshow(win_src, img_src_2);
}

void GrabCutClient::onMouse(int event, int x, int y, int flags,
                            void* userdata) {
  GrabCutClient* temp = reinterpret_cast<GrabCutClient*>(userdata);
  temp->onMouseMember(event, x, y, flags, userdata);
}

void GrabCutClient::onMouseMember(int event, int x, int y, int flags,
                                  void* userdata) {
  switch (event) {
    case EVENT_LBUTTONDOWN:
      rect_fgd.x = x;
      rect_fgd.y = y;
      rect_fgd.width = 1;
      rect_fgd.height = 1;
      break;
    case EVENT_MOUSEMOVE:
      if (flags && EVENT_FLAG_LBUTTON) {
        rect_fgd = Rect(Point(rect_fgd.x, rect_fgd.y), Point(x, y));
        redrawSrcImage();
      }
      break;
    case EVENT_LBUTTONUP:
      if (rect_fgd.width > 1 && rect_fgd.height > 1) {
        redrawSrcImage();
      }
      break;
    default:
      break;
  }
}

void GrabCutClient::iterateLabelMask() {
  // 用户框选的范围初始化为GC_PR_FGD，随着后续迭代计算，
  // 可能会有部分区域标记为GC_PR_BGD
  (mask_alpha(rect_fgd)).setTo(Scalar(GC_PR_FGD));

  /*这里写grabcut方法执行的代码*/
  // 初始化
  gc_method.initPixelsVec(img_src, mask_alpha);
  GrabCutGraph gc_graph = gc_method.initGraph();
  gc_method.clusterPixels();
  gc_method.fitTwoGMMs();
  cout << endl << " ===================>>>【初始化完成】" << endl;
  // 迭代中......
  for (int i = 0; i < i_iterate; i++) {
    cout << endl
         << " ===================>>>【第（" << (i + 1) << "）次迭代开始】"
         << endl;
    auto start = high_resolution_clock::now();  // 计时开始
    //////////////////////// 耗时部分 /////////////////////////
    //////////
    auto start_func = high_resolution_clock::now();  // 计时开始
    gc_method.updateTwoIndexMat();
    auto stop_func = high_resolution_clock::now();  //计时结束
    auto duration_func = duration_cast<seconds>(stop_func - start_func);
    cout << " ===================>>>【根据概率值更新索引 · 用时 "
         << duration_func.count() << " 秒】" << endl;
    //////////
    start_func = high_resolution_clock::now();  // 计时开始
    gc_method.fitTwoGMMs();
    stop_func = high_resolution_clock::now();  //计时结束
    duration_func = duration_cast<seconds>(stop_func - start_func);
    cout << " ===================>>>【拟合高斯模型 · 用时 "
         << duration_func.count() << " 秒】" << endl;
    //////////
    start_func = high_resolution_clock::now();  // 计时开始
    gc_method.updateMaskAlpha(gc_graph);
    stop_func = high_resolution_clock::now();  //计时结束
    duration_func = duration_cast<seconds>(stop_func - start_func);
    cout << " ===================>>>【执行图的最小割算法 · 用时 "
         << duration_func.count() << " 秒】" << endl;
    //////////
    mask_alpha = gc_method.getMaskAlpha();
    ///////////////////////////////
    auto stop = high_resolution_clock::now();  //计时结束
    auto duration = duration_cast<seconds>(stop - start);
    cout << " ===================>>>【第（" << (i + 1) << "）次迭代用时 "
         << duration.count() << " 秒】" << endl
         << endl;
  }
}

void GrabCutClient::showDstImage() {
  Mat mask_fgd;
  // 仅显示判定为GC_PR_FGD的区域
  compare(mask_alpha, GC_PR_FGD, mask_fgd, CMP_EQ);
  // 按照mask_alpha指示进行copy
  img_src.copyTo(img_dst, mask_fgd);
  namedWindow(win_dst, WINDOW_AUTOSIZE);
  waitKey(1);
  imshow(win_dst, img_dst);
}

void GrabCutClient::saveTwoImages(string s_date, string s_time_elapse) {
  string s_head = "results/";
  string s_stamp = (string)s_date + "_" + s_time_elapse + "_s_";
  if (b_opencv) {
    s_stamp += "OpenCV_" + to_string(i_iterate) + "_iter";
  } else {
    s_stamp += "Mine_" + to_string(i_iterate) + "_iter";
  }
  string s_ext = ".jpg";
  string s_src = s_head + s_stamp + "_src" + s_ext;
  string s_dst = s_head + s_stamp + "_dst" + s_ext;
  bool b_success_src = imwrite(s_src, img_src_2);
  bool b_success_dst = imwrite(s_dst, img_dst);
  if (b_success_src == false || b_success_dst == false) {
    cout << " ===================>>>【Failed to save the image!】" << endl;
  } else {
    cout << " ===================>>>【Images saved.】" << endl;
  }
}
