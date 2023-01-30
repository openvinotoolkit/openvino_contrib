# Java bindings for OpenVINO

[![Documentation Status](https://readthedocs.org/projects/openvino-contrib/badge/?version=latest)](https://openvino-contrib.readthedocs.io/en/latest/?badge=latest)

## Software Requirements
- OpenJDK (version depends on target OS)

### Linux
To install OpenJDK:

* Ubuntu systems:
```bash
sudo apt-get install -y default-jdk
```

### Build

Set environment OpenVINO variables:
```bash
source <openvino_install>/setupvars.sh
```

Use Gradle to build `openvino-x-x-x.jar` file with OpenVINO Java bindings:
```bash
cd openvino_contrib/modules/java_api
gradle build
```

### Import

Use `import org.intel.openvino.*;` for OpenVINO Java API 2.0 or `import org.intel.openvino.compatibility.*;` for deprecated API.
