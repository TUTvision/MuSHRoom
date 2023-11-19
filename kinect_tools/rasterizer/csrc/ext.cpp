#include <torch/extension.h>
#include "rasterize_points.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // module docstring
  m.doc() = "pybind11 compute_visibility_maps plugin";
  m.def("_rasterize", &RasterizePoints);
}
