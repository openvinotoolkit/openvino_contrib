# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

foreach(required_var IN ITEMS
        OV_GFX_PLUGINS_XML
        OV_TEMPLATE_PLUGIN_LIBRARY)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "${required_var} is required")
    endif()
endforeach()

get_filename_component(OV_GFX_PLUGINS_XML_DIR "${OV_GFX_PLUGINS_XML}" DIRECTORY)
file(MAKE_DIRECTORY "${OV_GFX_PLUGINS_XML_DIR}")

file(WRITE "${OV_GFX_PLUGINS_XML}"
    "<ie>\n"
    "    <plugins>\n"
    "        <plugin name=\"TEMPLATE\" location=\"${OV_TEMPLATE_PLUGIN_LIBRARY}\"/>\n"
    "    </plugins>\n"
    "</ie>\n")
