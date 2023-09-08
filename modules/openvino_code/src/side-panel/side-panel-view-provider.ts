import {
  CancellationToken,
  Disposable,
  Uri,
  Webview,
  WebviewView,
  WebviewViewProvider,
  WebviewViewResolveContext,
} from 'vscode';
import { SIDE_PANEL_VIEW_ID } from '../constants';
import { ISidePanelMessage } from '@shared/side-panel-message';
import { extensionState } from '../state';
import { IExtensionState } from '@shared/extension-state';
import { handleSidePanelMessage } from './side-panel-message-handler';

export class SidePanelViewProvider implements WebviewViewProvider {
  static viewId = SIDE_PANEL_VIEW_ID;

  private _view?: WebviewView;
  private _disposables: Disposable[] = [];

  constructor(
    private readonly _extensionUri: Uri,
    private readonly _isProductionMode: boolean
  ) {}

  resolveWebviewView(
    webviewView: WebviewView,
    _context: WebviewViewResolveContext<unknown>,
    _token: CancellationToken
  ): void | Thenable<void> {
    this._view = webviewView;
    this._view.webview.options = {
      enableScripts: true,
      localResourceRoots: this._getLocalResourceRoots(this._isProductionMode),
    };
    this._view.webview.html = this._getHtmlForWebview(this._view.webview, this._isProductionMode);

    this._view.onDidDispose(() => this._disposeView(), null, this._disposables);

    this._subscribeToWebviewMessages(this._view.webview);
    this._enablePostingToWebview(this._view.webview);
  }

  private _getLocalResourceRoots(isProductionMode: boolean): Uri[] {
    const localResourceRoots = [
      Uri.joinPath(this._extensionUri, 'out'),
      Uri.joinPath(this._extensionUri, 'side-panel-ui', 'dist'),
    ];

    if (!isProductionMode) {
      const devCodiconsDistUri = Uri.joinPath(this._extensionUri, 'node_modules', '@vscode', 'codicons', 'dist');
      localResourceRoots.push(devCodiconsDistUri);
    }

    return localResourceRoots;
  }

  private _disposeView(): void {
    while (this._disposables.length) {
      const disposable = this._disposables.pop();
      if (disposable) {
        disposable.dispose();
      }
    }
  }

  private _enablePostingToWebview(webview: Webview): void {
    const stateChangeListener = (state: IExtensionState) => {
      void webview.postMessage(state);
    };
    extensionState.subscribe(stateChangeListener);
    const stateChangeDisposable = new Disposable(() => {
      extensionState.unsubscribe(stateChangeListener);
    });
    this._disposables.push(stateChangeDisposable);
  }

  private _subscribeToWebviewMessages<M extends ISidePanelMessage>(webview: Webview): void {
    webview.onDidReceiveMessage(
      (message: M) => {
        handleSidePanelMessage(message, webview);
      },
      null,
      this._disposables
    );
  }

  private _getHtmlForWebview(webview: Webview, isProductionMode: boolean): string {
    return isProductionMode ? this._getProductionHtml(webview) : this._getDevelopmentHtml(webview);
  }

  private _getDevelopmentHtml(webview: Webview): string {
    const viteDevServerUrl = 'http://localhost:5173';
    const codiconsUri = webview.asWebviewUri(
      Uri.joinPath(this._extensionUri, 'node_modules', '@vscode', 'codicons', 'dist', 'codicon.css')
    );

    return `<!DOCTYPE html>
			<html lang="en">
        <head>
          <script type="module">
            import RefreshRuntime from "${viteDevServerUrl}/@react-refresh"
            RefreshRuntime.injectIntoGlobalHook(window)
            window.$RefreshReg$ = () => {}
            window.$RefreshSig$ = () => (type) => type
            window.__vite_plugin_react_preamble_installed__ = true
          </script>

          <script type="module" src="${viteDevServerUrl}/@vite/client"></script>
          <link href="${codiconsUri.toString()}" rel="stylesheet" />

          <meta charset="UTF-8">
        </head>
        <body>
          <div id="root">
            Vite dev server is not running. 
            <br />
            Run <code>npm run start:side-panel</code></div>
          <script type="module" src="${viteDevServerUrl}/src/main.tsx"></script>
        </body>
      </html>`;
  }

  private _getProductionHtml(webview: Webview): string {
    const sidePanelAssetsPathList = ['side-panel-ui', 'dist', 'assets'];
    const scriptUri = getUri(webview, this._extensionUri, [...sidePanelAssetsPathList, 'index.js']);
    const stylesUri = getUri(webview, this._extensionUri, [...sidePanelAssetsPathList, 'index.css']);
    const codiconUri = getUri(webview, this._extensionUri, [...sidePanelAssetsPathList, 'codicon.css']);

    const nonce = getNonce();

    return `<!DOCTYPE html>
			<html lang="en">
			<head>
				<meta charset="UTF-8">

				<!--
					Use a content security policy to only allow loading styles from our extension directory,
					and only allow scripts that have a specific nonce.
					(See the 'webview-sample' extension sample for img-src content security policy examples)
				-->
				<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${
          webview.cspSource
        }; script-src 'nonce-${nonce}'; img-src ${webview.cspSource}; font-src ${webview.cspSource}">

				<meta name="viewport" content="width=device-width, initial-scale=1.0">

        <link rel="stylesheet" type="text/css" href="${stylesUri.toString()}">
        <link rel="stylesheet" type="text/css" href="${codiconUri.toString()}">

				<title>OpenVINO Code - Side Panel</title>
			</head>
			<body>
        <div id="root"></div>
				<script nonce="${nonce}" src="${scriptUri.toString()}"></script>
			</body>
			</html>`;
  }
}

function getNonce(): string {
  let text = '';
  const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 32; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
}

function getUri(webview: Webview, extensionUri: Uri, pathList: string[]) {
  return webview.asWebviewUri(Uri.joinPath(extensionUri, ...pathList));
}
