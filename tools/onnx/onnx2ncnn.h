#include <expected.hpp>
#include <wmc_utils.h>

tl::expected<NcnnModel, std::string> onnx2ncnn(void **buf, size_t buflen);

