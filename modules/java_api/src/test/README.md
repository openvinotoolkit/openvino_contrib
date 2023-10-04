## Build

1. Download test data:
```bash
git clone https://github.com/openvinotoolkit/testdata
```

2. Run `gradle` to build a package and run tests:
```bash
gradle build -Prun_tests -DMODELS_PATH=/path/to/testdata -Ddevice=CPU --info
```
 
## Running tests in IntelliJ IDEA
This guide describes the steps to run tests for this module using IntelliJ IDEA. Before starting, ensure that you have IntelliJ IDEA 
installed and the project has been imported to IntelliJ IDEA.

### Downloading Test Data

Clone the following GitHub repository that contains the models needed to run the tests:
```shell
 git clone https://github.com/openvinotoolkit/testdata
```

### Creating the Gradle Configuration
1. Open the **Edit Configurations** dialog box from the IntelliJ IDEA toolbar, then select **Add New Configuration** > **Gradle** configuration from the dropdown menu

2. In the Name field, choose an appropriate name for the configuration, such as "AllTests", and enter the following command in the **Tasks and arguments** input box.
   ```shell
   test -Prun_tests -DMODELS_PATH=<path-to-testdata> -Ddevice=CPU
   ```

3. Click on the **Edit Environment Variables** button beside the **Environment Variables** input box and add the following environment variable: `INTEL_OPENVINO_DIR=<path-to-openvino_install>`

4. Select **OK** to save the configuration

### Running the Tests

Once the gradle configuration is created, run the tests using the following steps:

1. Select the Gradle configuration that was created in the Run/Debug configurations dropdown menu
2. Click on the **Run** button to run the tests
3. To run the tests in debug mode, click on the **Debug** button