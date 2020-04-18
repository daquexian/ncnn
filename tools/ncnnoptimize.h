#include <expected.hpp>
#include <wmc_utils.h>

tl::expected<NcnnModel, std::string> ncnnoptimize(const unsigned char *inparam, const unsigned char *inbin, int flag);
