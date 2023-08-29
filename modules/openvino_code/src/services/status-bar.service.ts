import { StatusBarAlignment, window, commands, ExtensionContext } from 'vscode';
import { COMMANDS, EXTENSION_DISPLAY_NAME } from '../constants';
import { extensionState } from '../state';
import { IExtensionState, ConnectionStatus } from '@shared/extension-state';
import { IExtensionComponent } from '../extension-component.interface';

class StatusBarContent {
  private _innerText: string;
  set innerText(value: StatusBarContent['_innerText']) {
    this._innerText = value;
    this._onUpdate?.(this.text);
  }

  private _suffixIcon: string | null = null;
  set suffixIcon(value: StatusBarContent['_suffixIcon']) {
    this._suffixIcon = value;
    this._onUpdate?.(this.text);
  }

  private _onUpdate: (text: string) => void;

  constructor(innerText: string, onUpdate: StatusBarContent['_onUpdate']) {
    this._onUpdate = onUpdate;
    this._innerText = innerText;
  }

  get text(): string {
    return `${this._innerText} ${this._suffixIcon ?? ''}`;
  }
}

class StatusBarService implements IExtensionComponent {
  private _statusBar = window.createStatusBarItem(StatusBarAlignment.Left, -1);

  private readonly _statusBarContent = new StatusBarContent(EXTENSION_DISPLAY_NAME, (text: string) => {
    this._statusBar.text = text;
  });

  private static readonly _icons = {
    loading: '$(loading~spin)',
    disconnect: '$(debug-disconnect)',
    pass: '$(pass)',
  };

  private static readonly _statusToIconMap: Record<ConnectionStatus, string> = {
    [ConnectionStatus.NOT_AVAILABLE]: StatusBarService._icons.disconnect,
    [ConnectionStatus.PENDING]: StatusBarService._icons.loading,
    [ConnectionStatus.AVAILABLE]: StatusBarService._icons.pass,
  };

  constructor() {
    this._statusBarContent.suffixIcon = StatusBarService._icons.disconnect;
    this._statusBar.tooltip = `Show ${EXTENSION_DISPLAY_NAME} Side Panel`;
    this._statusBar.command = COMMANDS.STATUS_BAR;
    this._statusBar.show();
    this._subscribeToStateChange();
  }

  private static _registerStatusBarCommand(context: ExtensionContext): void {
    const commandDisposable = commands.registerCommand(COMMANDS.STATUS_BAR, () => {
      void commands.executeCommand(COMMANDS.FOCUS_SIDE_PANEL);
    });
    context.subscriptions.push(commandDisposable);
  }

  private _subscribeToStateChange(): void {
    extensionState.subscribe((state) => {
      this._setSuffixIcon(state);
    });
  }

  private _setSuffixIcon({ isLoading, connectionStatus }: IExtensionState): void {
    const icon =
      isLoading || connectionStatus === ConnectionStatus.PENDING
        ? StatusBarService._icons.loading
        : StatusBarService._statusToIconMap[connectionStatus];

    this._statusBarContent.suffixIcon = icon;
  }

  activate(context: ExtensionContext): void {
    context.subscriptions.push(this._statusBar);
    StatusBarService._registerStatusBarCommand(context);
  }

  deactivate(): void {}
}

export const statusBarService = new StatusBarService();
