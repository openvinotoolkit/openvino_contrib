if(NOT DEFINED OUTPUT_FILE OR OUTPUT_FILE STREQUAL "")
    message(FATAL_ERROR "OUTPUT_FILE is required")
endif()

file(WRITE "${OUTPUT_FILE}"
"<ie>
    <plugins>
        <plugin name=\"GFX\" location=\"openvino_gfx_plugin\"></plugin>
        <plugin name=\"TEMPLATE\" location=\"openvino_template_plugin\"></plugin>
        <plugin name=\"AUTO\" location=\"openvino_auto_plugin\">
            <properties>
                <property key=\"MULTI_DEVICE_PRIORITIES\" value=\"GFX\"/>
                <property key=\"ENABLE_STARTUP_FALLBACK\" value=\"false\"/>
                <property key=\"ENABLE_RUNTIME_FALLBACK\" value=\"false\"/>
            </properties>
        </plugin>
    </plugins>
</ie>
")
