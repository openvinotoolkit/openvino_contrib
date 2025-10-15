import { ModelName } from '@shared/model';
import { DeviceName } from '@shared/device';
import { WorkspaceConfiguration, workspace } from 'vscode';
import { CONFIG_KEY } from './constants';

/**
 * Extension configuration should match `contributes.configuration` properties in package.json
 */
export type CustomConfiguration = {
  model: ModelName;
  device: DeviceName;
  serverUrl: string;
  serverRequestTimeout: number;
  streamInlineCompletion: boolean;
  fillInTheMiddleMode: boolean;
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
  minNewTokens: number;
  maxNewTokens: number;
  startToken: string;
  middleToken: string;
  endToken: string;
  stopToken: string;
} & {
  quoteStyle: string;
  docstringFormat: string;
};

export type ExtensionConfiguration = WorkspaceConfiguration & CustomConfiguration;

const hiddenConfigurations = {
  'openvinoCode.fillInTheMiddleMode': {
    order: 4,
    type: 'boolean',
    default: false,
    markdownDescription:
      'When checked, text before (above) and after (below) the cursor will be used for completion generation. When unckecked, only text before (above) the cursor will be used.',
  },
  'openvinoCode.startToken': {
    order: 7,
    type: 'string',
    default: '<fim_prefix>',
    markdownDescription:
      'String that is sent to server is in format: `{startToken}{text above cursor}{middleToken}{text below cursor if fillInTheMiddleMode=true}{endToken}`. Leave `startToken`, `middleToken`, or `endToken` empty if there is no special token for those placements.',
  },
  'openvinoCode.middleToken': {
    order: 8,
    type: 'string',
    default: '<fim_middle>',
    markdownDescription:
      'String that is sent to server is in format: `{startToken}{text above cursor}{middleToken}{text below cursor if fillInTheMiddleMode=true}{endToken}`. Leave `startToken`, `middleToken`, or `endToken` empty if there is no special token for those placements.',
  },
  'openvinoCode.endToken': {
    order: 9,
    type: 'string',
    default: '<fim_suffix>',
    markdownDescription:
      'String that is sent to server is in format: `{startToken}{text above cursor}{middleToken}{text below cursor if fillInTheMiddleMode=true}{endToken}`. Leave `startToken`, `middleToken`, or `endToken` empty if there is no special token for those placements.',
  },
  'openvinoCode.stopToken': {
    order: 10,
    type: 'string',
    default: '<|endoftext|>',
    description: '(Optional) Stop token.',
  },
};

const configurationDefaults: Partial<CustomConfiguration> = {
  fillInTheMiddleMode: hiddenConfigurations['openvinoCode.fillInTheMiddleMode'].default,
  startToken: hiddenConfigurations['openvinoCode.startToken'].default,
  middleToken: hiddenConfigurations['openvinoCode.middleToken'].default,
  endToken: hiddenConfigurations['openvinoCode.endToken'].default,
  stopToken: hiddenConfigurations['openvinoCode.stopToken'].default,
};

export const getConfig = () => ({
  ...configurationDefaults,
  ...(workspace.getConfiguration(CONFIG_KEY) as ExtensionConfiguration),
});
