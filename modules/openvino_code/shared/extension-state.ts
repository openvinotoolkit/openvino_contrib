import { ExtensionConfiguration } from '../src/configuration';
import { ServerState } from './server-state';

export enum ConnectionStatus {
  NOT_AVAILABLE = 'NOT_AVAILABLE',
  AVAILABLE = 'AVAILABLE',
  PENDING = 'PENDING',
}

export interface IExtensionState {
  isLoading: boolean;
  connectionStatus: ConnectionStatus;
  server: ServerState;
  get isServerAvailable(): boolean;
  get config(): ExtensionConfiguration;
}
