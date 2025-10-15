import { ISidePanelMessage, SidePanelMessageTypes } from '@shared/side-panel-message';
import { extensionState } from '../state';
import { Webview, commands } from 'vscode';
import { settingsService } from '../settings/settings.service';
import { COMMANDS } from '../constants';
import { ModelName } from '@shared/model';
import { DeviceName } from '@shared/device';

type SidePanelMessageHandlerType = (webview: Webview, payload?: ISidePanelMessage['payload']) => void;

const sidePanelMessageHandlers: Record<SidePanelMessageTypes, SidePanelMessageHandlerType> = {
  [SidePanelMessageTypes.GET_EXTENSION_STATE]: (webview) => void webview.postMessage(extensionState.state),
  [SidePanelMessageTypes.SETTINGS_CLICK]: () => settingsService.openSettings(),
  [SidePanelMessageTypes.MODEL_CHANGE]: (_, payload) =>
    settingsService.updateSetting('model', (payload as { modelName: ModelName }).modelName),
  [SidePanelMessageTypes.DEVICE_CHANGE]: (_, payload) =>
    settingsService.updateSetting('device', (payload as { deviceName: DeviceName }).deviceName),
  [SidePanelMessageTypes.START_SERVER_CLICK]: () => void commands.executeCommand(COMMANDS.START_SERVER_NATIVE),
  [SidePanelMessageTypes.STOP_SERVER_CLICK]: () => void commands.executeCommand(COMMANDS.STOP_SERVER_NATIVE),
  [SidePanelMessageTypes.SHOW_SERVER_LOG_CLICK]: () => void commands.executeCommand(COMMANDS.SHOW_SERVER_LOG),
  [SidePanelMessageTypes.SHOW_EXTENSION_LOG_CLICK]: () => void commands.executeCommand(COMMANDS.SHOW_EXTENSION_LOG),
  [SidePanelMessageTypes.CHECK_CONNECTION_CLICK]: () => void commands.executeCommand(COMMANDS.CHECK_CONNECTION),
  [SidePanelMessageTypes.GENERATE_COMPLETION_CLICK]: () =>
    void commands.executeCommand(COMMANDS.GENERATE_INLINE_COPMLETION),
};

export function handleSidePanelMessage<M extends ISidePanelMessage>(message: M, webview: Webview): void {
  const { type, payload } = message;
  const handler = sidePanelMessageHandlers[type];
  handler(webview, payload);
}
