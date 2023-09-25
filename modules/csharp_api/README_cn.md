# OpenVINO™ C# API 

 <img src="https://img.shields.io/badge/Framework-.NET6.0%2C%20.NET48-pink.svg">


简体中文| [English](README.md)

## 📚 简介

[OpenVINO™ ](www.openvino.ai)是一个用于优化和部署 AI 推理的开源工具包。

- 提升深度学习在计算机视觉、自动语音识别、自然语言处理和其他常见任务中的性能
- 使用流行框架（如TensorFlow，PyTorch等）训练的模型
- 减少资源需求，并在从边缘到云的一系列英特尔®平台上高效部署

&emsp;    该项目主要是基于OpenVINO™工具套件推出的 OpenVINO™ C# API，旨在推动  OpenVINO™  在C#平台的应用。

&emsp;    OpenVINO™ C# API 由于是基于 OpenVINO™  C API 开发，所支持的平台与OpenVINO™ 一致，具体信息可以参考 OpenVINO™。

## <img title="NuGet" src="https://s2.loli.net/2023/01/26/ks9BMwXaHqQnKZP.png" alt="" width="40"> NuGet Package

C# 支持 NuGet Package 方式安装程序包，在Linux、Window 等平台支持一站式安装使用，因此为了方便更多用户使用，目前发行了 Window 平台下使用的 NuGet Package ，方便大家使用。

| Package                 | Description                                                  | Link                                                         |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **OpenVINO.CSharp.win** | OpenVINO™ C# API core libraries，附带完整的OpenVINO 2023.1依赖库 | [![NuGet Gallery ](https://badge.fury.io/nu/OpenVINO.CSharp.win.svg)](https://www.nuget.org/packages/OpenVINO.CSharp.win/) |

## ⚙ 如何安装

以下文章提供了OpenVINO™ C# API在不同平台的安装方法，可以根据自己使用平台进行安装。

- [Windows](docs/cn/windows_install.md)

- [Linux](docs/cn/linux_install.md)

## 🏷开始使用

- **快速体验**

  [使用OpenVINO™ C# API部署Yolov8全系列模型](demos/yolov8/README_cn.md)

- **使用方法**

如果你不知道如何使用，通过下面代码简单了解使用方法。

```c#
namespace test 
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Core core = new Core();  // 初始化 Core 核心
            Model model = core.read_model("./model.xml");  // 读取模型文件
            CompiledModel compiled_model = core.compiled_model(model, "AUTO");  // 将模型加载到设备
            InferRequest infer_request = compiled_model.create_infer_request();  // 创建推理通道
            Tensor input_tensor = infer_request.get_tensor("images");  // 获取输入节点Tensor
            infer_request.infer();  // 模型推理
            Tensor output_tensor = infer_request.get_tensor("output0");  // 获取输出节点Tensor
            core.free();  // 清理 Core 非托管内存
        }
    }
}
```

项目中所封装的类、对象例如Core、Model、Tensor等，通过调用 C api 接口实现，具有非托管资源，需要调用**dispose()**方法处理，否则就会出现内存泄漏。

## 🗂 API 文档

如果想了解更多信息，可以参阅：[OpenVINO™ C# API API Documented](https://guojin-yan.github.io/OpenVINO-CSharp-API.docs/)



