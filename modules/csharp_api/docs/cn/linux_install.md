# OpenVINO C# API 在Linux 平台使用

&emsp;   由于目前 OpenVINO C# API 还在开发阶段，未生成相应的  NuGet Package， 因此此处基于 Ubuntu 20.04 系统，提供了相应的使用案例，方便大家在Linux系统上使用 OpenVINO C# API。

## 一、配置 .NET 环境

&emsp;    .NET 是一个免费的跨平台开源开发人员平台 ，用于构建多种应用程序。下面将演示 AIxBoard 如何在 Ubuntu 20.04 上安装 .NET环境，支持 .NET  Core 2.0-3.1 系列 以及.NET 5-8 系列 ，如果你的 AIxBoard 使用的是其他Linux系统，你可以参考[在 Linux 发行版上安装 .NET - .NET | Microsoft Learn](https://learn.microsoft.com/zh-cn/dotnet/core/install/linux)。

### 1. 添加 Microsoft 包存储库

&emsp;    使用 APT 进行安装可通过几个命令来完成。 安装 .NET 之前，请运行以下命令，将 Microsoft 包签名密钥添加到受信任密钥列表，并添加包存储库。

&emsp;    打开终端并运行以下命令：

```bash
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
rm packages-microsoft-prod.deb
```

&emsp;    下图为输入上面命令后控制台的输出：

<div align=center><span><img src="https://s2.loli.net/2023/08/01/2PGvUJbrR68axWt.png" height=300/></span></div>

### 2. 安装 SDK

&emsp;    .NET SDK 使你可以通过 .NET 开发应用。 如果安装 .NET SDK，则无需安装相应的运行时。 若要安装 .NET SDK，请运行以下命令：

```bash
sudo apt-get update
sudo apt-get install -y dotnet-sdk-3.1
```

&emsp;    下图为安装后控制台的输出：

<div align=center><span><img src="https://s2.loli.net/2023/08/08/tKY3oASu4Tf2dib.png" height=300/></span></div>


### 3. 测试安装

&emsp;    通过命令行可以检查 SDK 版本以及Runtime时版本。

```
dotnet --list-sdks
dotnet --list-runtimes
```

&emsp;    下图为输入测试命令后控制台的输出：

<div align=center><span><img src="https://s2.loli.net/2023/08/08/DQnli6Z3xOpVYvh.png" height=300/></span></div>

&emsp;    以上就是.NET环境的配置步骤，如果你的环境与本文不匹配，可以通过[.NET 文档 | Microsoft Learn](https://learn.microsoft.com/zh-cn/dotnet/) 获取更多安装步骤。

## 二、安装 OpenVINO C# API

&emsp;    OpenVINO™ 有两种安装方式: OpenVINO Runtime和OpenVINO Development Tools。OpenVINO Runtime包含用于在处理器设备上运行模型部署推理的核心库。OpenVINO Development Tools是一组用于处理OpenVINO和OpenVINO模型的工具，包括模型优化器、OpenVINO Runtime、模型下载器等。在此处我们只需要安装OpenVINO Runtime即可。

### 1. 下载 OpenVINO Runtime

&emsp;    访问[Download the Intel Distribution of OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?ENVIRONMENT=DEV_TOOLS&OP_SYSTEM=WINDOWS&VERSION=v_2023_0_1&DISTRIBUTION=PIP)页面，按照下面流程选择相应的安装选项，在下载页面，由于我们的设备使用的是**Ubuntu20.04**，因此下载时按照指定的编译版本下载即可。

<div align=center><span><img src="https://s2.loli.net/2023/08/01/BJ9SaVZmz8TUx4l.jpg" height=300/><img src="https://s2.loli.net/2023/08/01/GJbCHiSTtwdj791.jpg" height=200/></span></div>

### 2. 解压安装包

&emsp;    我们所下载的 OpenVINO Runtime 本质是一个C++依赖包，因此我们把它放到我们的系统目录下，这样在编译时会根据设置的系统变量获取依赖项。首先在系统文件夹下创建一个文件夹：

```bash
sudo mkdir -p /opt/intel
```

&emsp;    然后解压缩我们下载的安装文件，并将其移动到指定文件夹下：

```bash
tar -xvzf l_openvino_toolkit_ubuntu20_2023.0.1.11005.fa1c41994f3_x86_64.tgz
sudo mv l_openvino_toolkit_ubuntu20_2023.0.1.11005.fa1c41994f3_x86_64 /opt/intel/openvino_2022.3.0
```

### 3. 安装依赖

&emsp;    接下来我们需要安装 OpenVINO Runtime 所许雅的依赖项，通过命令行输入以下命令即可：

```bash
cd /opt/intel/openvino_2022.3.0/
sudo -E ./install_dependencies/install_openvino_dependencies.sh
```

<div align=center><span><img src="https://s2.loli.net/2023/08/01/B9ehCPf8KvXURFg.png" height=300/></span></div>

### 4. 配置环境变量

&emsp;    安装完成后，我们需要配置环境变量，以保证在调用时系统可以获取对应的文件，通过命令行输入以下命令即可：

```bash
source /opt/intel/openvino_2022.3.0/setupvars.sh
```

&emsp;    以上就是 OpenVINO Runtime 环境的配置步骤，如果你的环境与本文不匹配，可以通过[Install OpenVINO™ Runtime — OpenVINO™ documentation — Version(2023.0)](https://docs.openvino.ai/2023.0/openvino_docs_install_guides_install_runtime.html)获取更多安装步骤。

### 5. 添加 OpenVINO™ C# API 依赖

&emsp;    由于OpenVINO™ C# API当前正处于开发阶段，还未创建Linux版本的NuGet Package，因此需要通过下载项目源码以项目引用的方式使用。

- **下载源码**

  通过Git下载项目源码，新建一个Terminal，并输入以下命令克隆远程仓库，将该项目放置在项目同级目录下。

  ```
  git clone https://github.com/guojin-yan/OpenVINO-CSharp-API.git
  cd OpenVINO-CSharp-API
  ```

- **修改OpenVINO™ 依赖**

  由于项目源码的OpenVINO™ 依赖与本文设置不同，因此需要修改OpenVINO™ 依赖项的路径，主要通过修改``OpenVINO-CSharp-API/src/CSharpAPI/native_methods/ov_base.cs``文件即可，修改内容如下：

  ```
  private const string dll_extern = "./openvino2023.0/openvino_c.dll";
  ---修改为--->
  private const string dll_extern = "libopenvino_c.so";
  ```

- **添加项目依赖**

  在Terminal输入以下命令，即可将OpenVINO™ C# API添加到AlxBoard_deploy_yolov8项目引用中。

  ```shell
  dotnet add reference ./../OpenVINO-CSharp-API/src/CSharpAPI/CSharpAPI.csproj
  ```





