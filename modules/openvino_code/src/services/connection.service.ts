import { ExtensionContext, commands } from 'vscode';
import { backendService } from './backend.service';
import { extensionState } from '../state';
import { COMMANDS } from '../constants';
import { notificationService } from './notification.service';
import { IExtensionComponent } from '../extension-component.interface';
import { ConnectionStatus } from '@shared/extension-state';
import { ServerStatus } from '@shared/server-state';

class ConnectionService implements IExtensionComponent {
  private static readonly _checkIntervalMs = 60_000;

  private _intervalTimeout: NodeJS.Timeout | null = null;

  private async _checkConnection({ showNotification } = { showNotification: false }): Promise<void> {
    const wasAvailable = extensionState.get('isServerAvailable');
    extensionState.set('connectionStatus', ConnectionStatus.PENDING);

    const isAvailable = await backendService.healthCheck();

    extensionState.set('connectionStatus', isAvailable ? ConnectionStatus.AVAILABLE : ConnectionStatus.NOT_AVAILABLE);

    if (!isAvailable && (showNotification || wasAvailable)) {
      notificationService.showServerNotAvailableMessage(extensionState.state);
    }
  }

  private _registerCheckConnectionCommand(context: ExtensionContext): void {
    const commandDisposable = commands.registerCommand(COMMANDS.CHECK_CONNECTION, () => {
      void this._checkConnection({ showNotification: true });
    });
    context.subscriptions.push(commandDisposable);
  }

  private _enableConnectionCheck(): void {
    if (this._intervalTimeout) {
      console.error('Check connection interval already exists.');
      return;
    }

    this._intervalTimeout = setInterval(() => {
      void this._checkConnection();
    }, ConnectionService._checkIntervalMs);
  }

  private _disableConnectionCheck(): void {
    if (this._intervalTimeout) {
      clearInterval(this._intervalTimeout);
      this._intervalTimeout = null;
    }
  }

  activate(context: ExtensionContext): void {
    this._registerCheckConnectionCommand(context);

    // Enable interval connection check when server is started and disable when it is stopped
    extensionState.subscribe((state) => {
      if (!this._intervalTimeout && state.server.status === ServerStatus.STARTED) {
        this._enableConnectionCheck();
        void this._checkConnection({ showNotification: true });
        return;
      }
      if (this._intervalTimeout && state.server.status === ServerStatus.STOPPED) {
        this._disableConnectionCheck();
        void this._checkConnection({ showNotification: false });
      }
    });
  }

  deactivate(): void {
    this._disableConnectionCheck();
  }
}

export const connectionService = new ConnectionService();
