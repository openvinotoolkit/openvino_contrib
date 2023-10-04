import { ExtensionContext } from 'vscode';
import { statusBarService } from './services/status-bar.service';
import { extensionState } from './state';
import { docString } from './docstring';
import { connectionService } from './services/connection.service';
import { pythonServerService } from './python-server/python-server.service';
import { sidePanel } from './side-panel';
import { settingsService } from './settings/settings.service';
import { inlineCompletion } from './inline-completion';

const components = [
  extensionState,
  statusBarService,
  settingsService,
  inlineCompletion,
  docString,
  pythonServerService,
  sidePanel,
  connectionService,
];

export function activate(context: ExtensionContext): void {
  for (const component of components) {
    component.activate(context);
  }
}

export function deactivate() {
  for (const component of components) {
    component.deactivate();
  }
}
