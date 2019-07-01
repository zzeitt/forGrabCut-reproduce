#include "grabcut_client.h"

int main() {
  GrabCutClient gc_client("images/owl-test.jpg", false, 1 /*i_comp*/,
                          5 /*i_iterate*/);
  return 0;
}
