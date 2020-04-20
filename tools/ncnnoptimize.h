#include <expected.hpp>
#include <wmc_utils.h>

tl::expected<NcnnModel, std::string> ncnnoptimize(void **inparam, void **inbin, int flag);
