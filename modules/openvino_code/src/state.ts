import { EventEmitter } from 'stream';
import { ExtensionContext, workspace } from 'vscode';
import { ExtensionConfiguration } from './configuration';
import { CONFIG_KEY } from './constants';
import { IExtensionComponent } from './extension-component.interface';
import { IExtensionState, ConnectionStatus } from '@shared/extension-state';
import { INITIAL_SERVER_STATE, ServerStatus } from '@shared/server-state';

class ExtensionState implements IExtensionComponent {
  private readonly _state: IExtensionState = {
    isLoading: false,
    connectionStatus: ConnectionStatus.NOT_AVAILABLE,
    server: INITIAL_SERVER_STATE,
    get isServerAvailable(): boolean {
      return this.server.status === ServerStatus.STARTED && this.connectionStatus === ConnectionStatus.AVAILABLE;
    },
    get config(): ExtensionConfiguration {
      return workspace.getConfiguration(CONFIG_KEY) as ExtensionConfiguration;
    },
  };

  private _emitter = new EventEmitter();

  private static readonly _stateChangedEvent = 'ExtensionState:stateChanged';

  private _extensionContext: ExtensionContext | null = null;

  get state(): IExtensionState {
    return this._state;
  }

  get config(): ExtensionConfiguration {
    return this._state.config;
  }

  activate(extensionContext: ExtensionContext): void {
    // Might be used to store configuration in `extensionContext.globalState`
    this._extensionContext = extensionContext;
  }

  set<K extends keyof IExtensionState>(key: K, value: IExtensionState[K]): void {
    this._state[key] = value;
    this._emitter.emit(ExtensionState._stateChangedEvent, this._state);
  }

  get<K extends keyof IExtensionState>(key: K): IExtensionState[K] {
    return this._state[key];
  }

  subscribe(listener: (state: IExtensionState) => void): void {
    this._emitter.on(ExtensionState._stateChangedEvent, listener);
  }

  unsubscribe(listener: (state: IExtensionState) => void): void {
    this._emitter.removeListener(ExtensionState._stateChangedEvent, listener);
  }

  deactivate(): void {
    this._emitter.removeAllListeners();
  }
}

export const extensionState = new ExtensionState();
