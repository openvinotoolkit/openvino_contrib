
## Build and install `openvino-langchain` from source

### openvino-genai-node

Follow these [build instructions](https://github.com/openvinotoolkit/openvino.genai/blob/master/src/js/BUILD.md#build-openvino-genai-nodejs-bindings-preview) to build OpenVINO™ GenAI Node.js bindings. Additionally, run the following commands:

```bash
npm i
rm -r ~/openvino.genai/src/js/node_modules/openvino-node/bin/*
cd ~/openvino.genai/src/js/node_modules/openvino-node/
cp -r ~/openvino/src/bindings/js/node/bin/* .
```

### openvino-node

To build OpenVINO™ Node.js bindings refer to this [developer documentation](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/docs/README.md#openvino-node-package-developer-documentation). 

> **NOTE**:  Ensure you specify the correct installation directory for `openvino_node`. Use the following flags when configuring:
> 
> ```bash
> -DCPACK_GENERATOR=NPM
> -DCMAKE_INSTALL_PREFIX="../src/bindings/js/node/bin"
> ```


### openvino-langchain

Clone the repository and install dependencies:

```bash
git clone https://github.com/intel-sandbox/openvino-langchain.git
cd openvino-langchain/
npm i
```
