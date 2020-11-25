# An overview of the openvino_contrib modules

This list gives an overview of all modules available inside the contrib repository. To turn off building one of these module repositories, set the names in bold below to

```sh
$ cmake -DIE_EXTRA_MODULES=<openvino_contrib>/modules -DBUILD_<module name>=OFF <openvino_source_directory>
```

* [**java_api**](./java_api): Inference Engine Java API -- provides Java wrappers for Inference Engine public API.
* [**mo_pytorch**](./mo_pytorch): PyTorch extensions for Model Optimizer -- native PyTorch to OpenVINO IR converter
