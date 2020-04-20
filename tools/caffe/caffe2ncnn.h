#include <expected.hpp>
#include <wmc_utils.h>

tl::expected<NcnnModel, std::string> caffe2ncnn(void** txt_buf,
                                                const size_t txt_len,
                                                void** model_buf,
                                                const size_t model_len);
