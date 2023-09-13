import { ModelName } from '@shared/model';
import { WorkspaceConfiguration } from 'vscode';

/**
 * Extension configuration should match `contributes.configuration` properties in package.json
 */
export type CustomConfiguration = {
  model: ModelName;
  serverUrl: string;
  serverRequestTimeout: number;
  streamInlineCompletion: boolean;
  fillInTheMiddleMode: boolean;
  temperature: number;
  topK: number;
  topP: number;
  minNewTokens: number;
  maxNewTokens: number;
  startToken: string;
  middleToken: string;
  endToken: string;
  stopToken: string;
} & {
  quoteStyle?: string;
  docstringFormat?: string;
};

export type ExtensionConfiguration = WorkspaceConfiguration & CustomConfiguration;
