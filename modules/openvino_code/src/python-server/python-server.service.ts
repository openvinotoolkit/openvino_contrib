import { ExtensionContext, commands } from 'vscode';

import { COMMANDS, EXTENSION_PACKAGE, EXTENSION_SERVER_DISPLAY_NAME } from '../constants';
import { IExtensionComponent } from '../extension-component.interface';
import { NativePythonServerRunner } from './python-server-runner';
import { extensionState } from '../state';

class PythonServerService implements IExtensionComponent {
  private _pythonServer = new NativePythonServerRunner();

  activate(context: ExtensionContext): void {
    const startCommandDisposable = commands.registerCommand(COMMANDS.START_SERVER_NATIVE, async () => {
      void commands.executeCommand(COMMANDS.SHOW_SERVER_LOG);
      await this._pythonServer.start();
    });

    const stopCommandDisposable = commands.registerCommand(COMMANDS.STOP_SERVER_NATIVE, () =>
      this._pythonServer.stop()
    );

    const showLogCommandDisposable = commands.registerCommand(COMMANDS.SHOW_SERVER_LOG, () => {
      void commands.executeCommand(
        `workbench.action.output.show.${EXTENSION_PACKAGE.fullName}.${EXTENSION_SERVER_DISPLAY_NAME}`
      );
    });

    const stateSubscriptionDisposable = this._pythonServer.stateReporter.subscribeToStateChange((serverState) => {
      extensionState.set('server', serverState);
    });

    context.subscriptions.push(
      startCommandDisposable,
      stopCommandDisposable,
      showLogCommandDisposable,
      stateSubscriptionDisposable
    );
  }

  deactivate() {
    this._pythonServer.stop();
  }
}

export const pythonServerService = new PythonServerService();
