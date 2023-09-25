# Using the OpenVINO C # API on the Linux

&emsp;   Due to the fact that the OpenVINO C # API is still in the development stage and no corresponding NuGet Package has been generated, corresponding use cases are provided based on the Ubuntu 20.04 system to facilitate everyone's use of the OpenVINO C # API on Linux systems.

## Ⅰ. Install. NET

&emsp;&emsp;    . NET is a free cross platform open source developer platform for building multiple applications. The following will demonstrate how AIxBoard can install the. NET environment on Ubuntu 20.04, supporting the. NET Core 2.0-3.1 series and. NET 5-8 series. If your AIxBoard is using another Linux system, you can refer to [Install .NET on Linux distributions - .NET | Microsoft Learn](https://learn.microsoft.com/en-us/dotnet/core/install/linux)

### 1. Add Microsoft Package Repository

&emsp;    The installation using APT can be completed through several commands. Before installing. NET, please run the following command to add the Microsoft package signing key to the trusted key list and add the package repository.

&emsp;    Open the terminal and run the following command:

```bash
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
rm packages-microsoft-prod.deb
```

&emsp;    The following figure shows the output of the console after entering the above command:

<div align=center><span><img src="https://s2.loli.net/2023/08/01/2PGvUJbrR68axWt.png" height=300/></span></div>

### 2. Install SDK

&emsp;    The. NET SDK allows you to develop applications through. NET. If you install the. NET SDK, you do not need to install the corresponding runtime. To install the. NET SDK, run the following command:

```bash
sudo apt-get update
sudo apt-get install -y dotnet-sdk-3.1
```

&emsp;    The following figure shows the output of the console after entering the above command:

<div align=center><span><img src="https://s2.loli.net/2023/08/08/tKY3oASu4Tf2dib.png" height=300/></span></div>


### 3. Test installation

&emsp;    You can check the SDK version and runtime version through the command line.

```
dotnet --list-sdks
dotnet --list-runtimes
```

&emsp;    The following figure shows the output of the console after entering the above command:

<div align=center><span><img src="https://s2.loli.net/2023/08/08/DQnli6Z3xOpVYvh.png" height=300/></span></div>

&emsp;    The above are the configuration steps for the. NET environment. If your environment does not match this article, you can obtain more installation steps through [.NET documentation | Microsoft Learn](https://learn.microsoft.com/en-us/dotnet/).

## Ⅲ. Install OpenVINO Runtime

&emsp;    OpenVINO™  have two installation methods: OpenVINO Runtime and OpenVINO Development Tools. The OpenVINO Runtime contains a core library for running model deployment inference on processor devices. OpenVINO Development Tools is a set of tools used to process OpenVINO and OpenVINO models, including model optimizer, OpenVINO runtime, model downloader, and more. We only need to install OpenVINO Runtime here.

### 1. Download OpenVINO Runtime

&emsp;   Visit the [Download the Intel Distribution of OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?ENVIRONMENT=DEV_TOOLS&OP_SYSTEM=WINDOWS&VERSION=v_2023_0_1&DISTRIBUTION=PIP) page and follow the process below to select the corresponding installation options. On the download page, as our device is using **Ubuntu 20.04 **, download according to the specified compiled version.

<div align=center><span><img src="https://s2.loli.net/2023/08/01/BJ9SaVZmz8TUx4l.jpg" height=300/><img src="https://s2.loli.net/2023/08/01/GJbCHiSTtwdj791.jpg" height=200/></span></div>

### 2. Unzip installation package

&emsp;    The OpenVINO Runtime we downloaded is essentially a C++dependency package, so we placed it in our system directory so that dependencies can be obtained during compilation based on the set system variables. First, create a folder under the system folder:

```bash
sudo mkdir -p /opt/intel
```

&emsp;    Then extract the installation files we downloaded and move them to the specified folder:

```bash
tar -xvzf l_openvino_toolkit_ubuntu20_2023.0.1.11005.fa1c41994f3_x86_64.tgz
sudo mv l_openvino_toolkit_ubuntu20_2023.0.1.11005.fa1c41994f3_x86_64 /opt/intel/openvino_2022.3.0
```

### 3. Installation dependencies

&emsp;    Next, we need to install the dependencies required by the OpenVINO Runtime. Enter the following command from the command line:

```bash
cd /opt/intel/openvino_2022.3.0/
sudo -E ./install_dependencies/install_openvino_dependencies.sh
```

<div align=center><span><img src="https://s2.loli.net/2023/08/01/B9ehCPf8KvXURFg.png" height=300/></span></div>

### 4. Configure environment variables

&emsp;    After the installation is completed, we need to configure the environment variables to ensure that the system can obtain the corresponding files when calling. Enter the following command from the command line:

```bash
source /opt/intel/openvino_2022.3.0/setupvars.sh
```

&emsp;    The above are the configuration steps for the OpenVINO Runtime environment. If your environment does not match this article, you can obtain more installation steps through [Install OpenVINO™ Runtime — OpenVINO™ documentation — Version(2023.0)](https://docs.openvino.ai/2023.0/openvino_docs_install_guides_install_runtime.html).

### 5. Add OpenVINO™ C# API Dependency

&emsp;    Due to the fact that OpenVINO™ C# API is currently in the development phase and has not yet created a Linux version of NuGet Package, 

- **Download source code**

  Due to OpenVINO ™  C # API is currently in the development stage and has not yet created a Linux version of NuGet Package. Therefore, it needs to be used by downloading the project source code as a project reference.

  ```
  git clone https://github.com/guojin-yan/OpenVINO-CSharp-API.git
  cd OpenVINO-CSharp-API
  ```

  

- **Modify OpenVINO ™ Dependency**

  Due to the OpenVINO™ dependency of the project source code being different from the settings in this article, it is necessary to modify the path of the OpenVINO™ dependency, mainly by modifying the``OpenVINO-CSharp-API/src/CSharpAPI/native_methods/ov_base.cs``. The modifications are as follows:

  ```
  private const string dll_extern = "./openvino2023.0/openvino_c.dll";
  ---Modify to--->
  private const string dll_extern = "libopenvino_c.so";
  ```

- **Add Project Dependency**

  Enter the following command in Terminal to add OpenVINO™ C# API to AlxBoard_ Deploy_ Yolov8 project reference.

  ```
  dotnet add reference ./../OpenVINO-CSharp-API/src/CSharpAPI/CSharpAPI.csproj
  ```

