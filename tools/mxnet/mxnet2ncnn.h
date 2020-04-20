#include <expected.hpp>
#include <wmc_utils.h>

tl::expected<NcnnModel, std::string> mxnet2ncnn(void** nodes_buf,
                                                const size_t nodes_len,
                                                void** params_buf,
                                                const size_t params_len);
