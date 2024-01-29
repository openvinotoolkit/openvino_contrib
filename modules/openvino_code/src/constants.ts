export const EXTENSION_PACKAGE = {
  publisher: 'OpenVINO',
  name: 'openvino-code-completion',
  get fullName(): string {
    return `${this.publisher}.${this.name}`;
  },
};

export const EXTENSION_DISPLAY_NAME = 'OpenVINO Code';
export const EXTENSION_SERVER_DISPLAY_NAME = 'OpenVINO Code Server';

export const CONFIG_KEY = 'openvinoCode';

export const SIDE_PANEL_VIEW_ID = 'openvino-code-side-panel';

export const COMMANDS = {
  STATUS_BAR: 'openvinoCode.statusBar',
  FOCUS_SIDE_PANEL: `${SIDE_PANEL_VIEW_ID}.focus`,
  OPEN_SETTINGS: 'openvinoCode.openSettings',
  GENERATE_INLINE_COPMLETION: 'openvinoCode.generateInlineCompletion',
  GENERATE_INLINE_COPMLETION_TAB: 'openvinoCode.generateInlineCompletionTab',
  ACCEPT_INLINE_COMPLETION: 'openvinoCode.acceptInlineCompletion',
  GENERATE_DOC_STRING: 'openvinoCode.generateDocstring',
  CHECK_CONNECTION: 'openvinoCode.checkConnection',
  START_SERVER_NATIVE: 'openvinoCode.startServerNative',
  STOP_SERVER_NATIVE: 'openvinoCode.stopServerNative',
  SHOW_SERVER_LOG: 'openvinoCode.showServerLog',
  SHOW_EXTENSION_LOG: 'openvinoCode.showExtensionLog',
  STOP_GENERATION: 'openvinoCode.stopGeneration',
};

export const EXTENSION_CONTEXT_STATE = {
  GENERATING: 'openvinoCode.generating',
};
