#include <stdexcept>
#include <string>

namespace ov {
namespace test {
void set_device_suffix(const std::string& suffix) {
    if (!suffix.empty()) {
        throw std::runtime_error("The suffix can't be used for TEMPLATE device!");
    }
}
}  // namespace test
}  // namespace ov

