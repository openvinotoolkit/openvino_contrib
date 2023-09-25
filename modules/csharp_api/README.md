# OpenVINO™ C# API 

 <img src="https://img.shields.io/badge/Framework-.NET6.0%2C%20.NET48-pink.svg">

[简体中文](README_cn.md) | English

## 📚 What is OpenVINO™ C# API ?

[OpenVINO™](www.openvino.ai)  is an open-source toolkit for optimizing and deploying AI inference.

- Boost deep learning performance in computer vision, automatic speech recognition, natural language processing and other common tasks
- Use models trained with popular frameworks like TensorFlow, PyTorch and more
- Reduce resource demands and efficiently deploy on a range of Intel® platforms from edge to cloud

&emsp;    This project is mainly based on OpenVINO ™ OpenVINO launched by tool kit ™  C # API, aimed at driving OpenVINO ™   Application on the C # platform.

&emsp;    OpenVINO ™  The C # API is based on OpenVINO ™   C API development, supported platforms and OpenVINO ™  Consistent, please refer to OpenVINO for specific information ™。

## <img title="NuGet" src="https://s2.loli.net/2023/01/26/ks9BMwXaHqQnKZP.png" alt="" width="40"> NuGet Package

&emsp;    C # supports NuGet Package installation and one-stop installation on platforms such as Linux and Window. Therefore, in order to facilitate more users, a NuGet Package for use on the Window platform has been released for the convenience of everyone.

| Package                 | Description                                                  | Link                                                         |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **OpenVINO.CSharp.win** | OpenVINO™ C# API core libraries，comes with a complete OpenVINO 2023.0 dependency library | [![NuGet Gallery ](https://badge.fury.io/nu/OpenVINO.CSharp.win.svg)](https://www.nuget.org/packages/OpenVINO.CSharp.win/) |

## ⚙ How to install OpenVINO™ C# API?

The following article provides installation methods for OpenVINO™ C# API on different platforms, which can be installed according to your own platform.

- [Windows](docs/en/windows_install.md)

- [Linux](docs/en/linux_install.md)

## 🏷How to use OpenVINO™ C# API?

- **Quick start**
  - [Deploying the Yolov8 full series model using OpenVINO™ C# API](demos/yolov8/README.md)

- **Simple usage**

If you don't know how to use it, simply understand the usage method through the following code.

```c#
namespace test 
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Core core = new Core();
            Model model = core.read_model("./model.xml");
            CompiledModel compiled_model = core.compiled_model(model, "AUTO"); 
            InferRequest infer_request = compiled_model.create_infer_request(); 
            Tensor input_tensor = infer_request.get_tensor("images"); 
            infer_request.infer(); 
            Tensor output_tensor = infer_request.get_tensor("output0"); 
            core.free(); 
        }
    }
}
```

The classes and objects encapsulated in the project, such as Core, Model, Tensor, etc., are implemented by calling the C API interface and have unmanaged resources. They need to be handled by calling the **dispose() ** method, otherwise memory leakage may occur.

## 🗂 API Reference

If you want to learn more information, you can refer to: [OpenVINO™ C# API API Documented](https://guojin-yan.github.io/OpenVINO-CSharp-API.docs/)
