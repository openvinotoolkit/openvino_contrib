# Java bindings for OpenVINO

## Software Requirements
- OpenJDK (version depends on target OS)

### Linux
* Ubuntu systems:
    ```bash
    sudo apt-get install -y default-jdk
    ```

### macOS
```bash
brew install openjdk
```

## Build

To build OpenVINO so that it includes this module, use the following CMake command:
```shell
cd <openvino_build>
cmake -DBUILD_java_api=ON -DOPENVINO_EXTRA_MODULES=<openvino_contrib>/modules <openvino_source_directory>
cmake --build . -j8
```

Set OpenVINO environment variables:
```bash
source <openvino_install>/setupvars.sh
```

Use Gradle to build `openvino-x-x-x.jar` file with OpenVINO Java bindings:
```bash
cd <openvino_contrib>/modules/java_api
gradle build
```

## Import

Use `import org.intel.openvino.*;` for OpenVINO Java API.

## Set up the development environment

### Import to IntelliJ IDEA

- Install and enable the **Gradle** IntelliJ Plugin by navigating to **Settings** > **Plugins**. Search for the
  Gradle plugin and install it, if not already installed.
- Clone the repository
  ```shell
  git clone https://github.com/openvinotoolkit/openvino_contrib.git
  ```
- To import the project into IntelliJ IDEA, select **File** > **Open** and locate the  java api module in `<openvino_contrib>/modules/java_api`.

See [here](src/test/README.md) for instructions on running tests.