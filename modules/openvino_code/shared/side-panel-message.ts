const sidePanelMessagePrefix = 'side-panel.message';

export enum SidePanelMessageTypes {
  GET_EXTENSION_STATE = `${sidePanelMessagePrefix}.getExtensionState`,
  START_SERVER_CLICK = `${sidePanelMessagePrefix}.startServerClick`,
  STOP_SERVER_CLICK = `${sidePanelMessagePrefix}.stopServerClick`,
  SHOW_SERVER_LOG_CLICK = `${sidePanelMessagePrefix}.showServerLogClick`,
  SHOW_EXTENSION_LOG_CLICK = `${sidePanelMessagePrefix}.showExtensionLogClick`,
  CHECK_CONNECTION_CLICK = `${sidePanelMessagePrefix}.checkConnectionClick`,
  GENERATE_COMPLETION_CLICK = `${sidePanelMessagePrefix}.generateCompletionClick`,
  SETTINGS_CLICK = `${sidePanelMessagePrefix}.settingsClick`,
  MODEL_CHANGE = `${sidePanelMessagePrefix}.modelChange`,
  DEVICE_CHANGE = `${sidePanelMessagePrefix}.deviceChange`,
}

export interface ISidePanelMessage<P = unknown> {
  type: SidePanelMessageTypes;
  payload?: P;
}
