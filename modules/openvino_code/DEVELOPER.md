# OpenVINO Code - VSCode extension for AI code completion with OpenVINO™

VSCode extension for helping developers writing code with AI code assistant. OpenVINO Code is working with Large Language Model for Code (Code LLM) deployed on local or remote server.

## Installing Extension

VSCode extension can be installed from built `*.vsix` file:

1. Open `Extensions` side bar in VSCode.
2. Click on the menu icon (three dots menu icon aka "meatballs" icon) in the top right corner of Extensions side panel.
3. Select "Instal from VSIX..." option and select extension file.

For instructions on how to build extension `vsix` file please refer to the [Build Extension](#build-extension) section.

## Extension Configuration

To work with extension you should configure endpoint to server with Code LLM where requests will be sent:

1. Open extension settings.
2. Fill `Server URL` parameter with server endpoint URL.

For instructions on how to start server locally please refer to the [server README.md](./server/README.md).

Also in extension settings you can configure special tokens.

## Working with Extension

TDB

1. Create a new python file
2. Try typing `def main():`
3. Press shortcut buttons (TBD) for code completion

### Checking output

You can see input to and output from the code generation API:

1. Open VSCode `OUTPUT` panel
2. Select extension output source from the dropdown menu

## Developing

> **Prerequisite:** You should have `Node.js` installed (v16 and above).

#### Install dependencies

To install dependencies run the following command from the project root directory:

```
npm install
```

#### Run Extension from Source & Debugging

Open `Run and Debug` side bar in VSCode and click `Launch Extension` (or press `F5`).

#### Build Extension

To build extension and generate `*.vsix` file for further installation in VSCode, run the following command:

```
npm run vsce:package
```

#### Linting

To perform linting with `ESLint`, execute the following command:

```
npm run lint
```

#### Testing

TBD
