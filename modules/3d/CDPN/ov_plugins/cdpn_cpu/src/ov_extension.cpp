// Extension entry point - registers all CDPN custom ops with OV runtime.

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>

#include "cdpn_preprocess.hpp"
#include "cdpn_coord_denorm.hpp"
#include "cdpn_trans_decode.hpp"
#include "cdpn_pnp_solve.hpp"

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        std::make_shared<ov::OpExtension<CdpnExtension::CdpnPreprocess>>(),
        std::make_shared<ov::OpExtension<CdpnExtension::CdpnCoordDenorm>>(),
        std::make_shared<ov::OpExtension<CdpnExtension::CdpnTransDecode>>(),
        std::make_shared<ov::OpExtension<CdpnExtension::CdpnPnpSolve>>(),
    }));
