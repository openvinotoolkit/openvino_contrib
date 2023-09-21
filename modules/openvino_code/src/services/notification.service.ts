import { commands, window } from 'vscode';
import { COMMANDS, EXTENSION_DISPLAY_NAME } from '../constants';
import { extensionState } from '../state';
import { settingsService } from '../settings/settings.service';
import { IExtensionState } from '@shared/extension-state';
import { ServerStatus } from '@shared/server-state';

class NotificationService {
  private static readonly _actions = {
    startServer: 'Start Server',
    configureSettings: 'Configure in Settings',
    tryAgain: 'Check Connection',
  };

  showServerNotAvailableMessage(state: IExtensionState): void {
    if (state.server.status === ServerStatus.STOPPED) {
      const message = `${EXTENSION_DISPLAY_NAME}: Server is stopped.`;
      const { startServer } = NotificationService._actions;
      void window.showWarningMessage(message, startServer).then((selection) => {
        if (!selection) {
          return;
        }
        if (selection === startServer) {
          void commands.executeCommand(COMMANDS.START_SERVER_NATIVE);
        }
      });
      return;
    }

    const message = `${EXTENSION_DISPLAY_NAME}: Server (${extensionState.config.serverUrl}) is not available.`;
    const { configureSettings, tryAgain } = NotificationService._actions;

    void window.showWarningMessage(message, configureSettings, tryAgain).then((selection) => {
      if (!selection) {
        return;
      }
      if (selection === configureSettings) {
        settingsService.openSettings('serverUrl');
      } else if (selection === tryAgain) {
        void commands.executeCommand(COMMANDS.CHECK_CONNECTION);
      }
    });
  }

  showWarningMessage(message: string): void {
    const warningMessage = `${EXTENSION_DISPLAY_NAME}: ${message}`;
    void window.showWarningMessage(warningMessage);
  }

  showRequestTimeoutMessage(): void {
    const message = `${EXTENSION_DISPLAY_NAME}: Request to server was aborted due to timeout.`;
    const { configureSettings } = NotificationService._actions;

    void window.showErrorMessage(message, configureSettings).then((selection) => {
      if (selection === configureSettings) {
        settingsService.openSettings('serverRequestTimeout');
      }
    });
  }
}

export const notificationService = new NotificationService();
