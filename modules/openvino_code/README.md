# OpenVINO Code - VSCode extension for AI code completion with OpenVINO™

VSCode extension for helping developers writing code with AI code assistant.
OpenVINO Code is working with Large Language Model for Code (Code LLM) deployed on local server
or remote server using [Remote Explorer](https://marketplace.visualstudio.com/items?itemName=ms-vscode.remote-explorer).

OpenVINO Code provides the following features:

- Inline Code Completion
- Summarization via Docstring

## Working with Extension

### Starting Server

On the extension side panel, choose your preferred model from the available options. 
The features supported by the selected model, which will be displayed under the model selector.

Once the server is up and running, you can access instructions for utilizing various functions available with selected model on the sidebar of the extension.
Now you can check the server's status and connection status. 
Additionally, connection status shown on the VSCode Status Bar.
To check the connection manually, use the `Check Connection` button located on the side panel. 

### Code Completion

[Code Completion](https://github.com/openvinotoolkit/openvino_contrib/assets/51917466/515c38c9-6a16-4884-9c84-8f096be1fd27)

1. Create a new Python file or open an existing one.
1. Type `def main():` or place the cursor where you'd like code suggestions to be generated.
1. Press the keyboard shortcut `Ctrl+Alt+Space`/`Cmd+Alt+Space` or click the `Generate Code Completion` button located in the side panel.
1. Use the `Tab` key to accept the entire suggestion or `Ctrl`+`Right Arrow` to accept it word by word. To decline the suggestion, press `Esc`.

You can customize the length of the generated code by adjusting `Max New Tokens` and `Min New Tokens` parameters in the extension settings. 
The number of generated tokens is also influenced by the `Server Request Timeout` setting.

To enable streaming generation mode, check the `Stream Inline Completion` checkbox in the extension settings.
This mode allows you to immediately receive model output and avoid problems with server response timeouts.

### Summarization via Docstring Generation

[Summarization via Docstring Generation](https://github.com/openvinotoolkit/openvino_contrib/assets/51917466/f56d2bef-a6be-407c-ae58-f8915db51c74)

To generate function docstring start typing `"""` or `'''` right under function signature and choose `Generate Docstring`.
You can select the desired type of quotes in the extension settings.

The model can generate docstring in Code Completion mode, but in this case it is impossible to control the result. 
In the docstring generation mode, various popular templates are available in the settings that will guide the model output.

### Monitoring Extension Output

To examine the input and output from the code generation API, follow these steps:

1. Open OpenVINO Code Side Panel
1. Choose between two options: `Show Server Log` or `Show Extension Log`.
