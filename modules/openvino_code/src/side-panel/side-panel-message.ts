const sidePanelMessagePrefix = 'side-panel.message';

export enum SidePanelMessageTypes {
  GET_EXTENSION_STATE = `${sidePanelMessagePrefix}.getExtensionState`,
  START_SERVER_CLICK = `${sidePanelMessagePrefix}.startServerClick`,
  STOP_SERVER_CLICK = `${sidePanelMessagePrefix}.stopServerClick`,
  SHOW_SERVER_LOG_CLICK = `${sidePanelMessagePrefix}.showServerLogClick`,
  GENERATE_COMPLETION_CLICK = `${sidePanelMessagePrefix}.generateCompletionClick`,
  SETTINGS_CLICK = `${sidePanelMessagePrefix}.settingsClick`,
}

export interface ISidePanelMessage<P = unknown> {
  type: SidePanelMessageTypes;
  payload?: P;
}
