#include <expected.hpp>
#include <wmc_utils.h>

tl::expected<NcnnModel, std::string> mxnet2ncnn(const std::string &nodes_str, const std::string &params_str);
