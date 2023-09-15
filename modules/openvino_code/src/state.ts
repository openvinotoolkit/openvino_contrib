import { EventEmitter } from 'stream';
import { ExtensionContext, workspace } from 'vscode';
import { ExtensionConfiguration, getConfig } from './configuration';
import { CONFIG_KEY } from './constants';
import { IExtensionComponent } from './extension-component.interface';
import { IExtensionState, ConnectionStatus } from '@shared/extension-state';
import { INITIAL_SERVER_STATE, ServerStatus } from '@shared/server-state';
import { MODEL_SUPPORTED_FEATURES } from '@shared/model';
import { Features } from '@shared/features';

class ExtensionState implements IExtensionComponent {
  private readonly _state: IExtensionState = {
    isLoading: false,
    connectionStatus: ConnectionStatus.NOT_AVAILABLE,
    server: INITIAL_SERVER_STATE,
    get isServerAvailable(): boolean {
      return this.server.status === ServerStatus.STARTED && this.connectionStatus === ConnectionStatus.AVAILABLE;
    },
    get config(): ExtensionConfiguration {
      return getConfig();
    },
    features: {
      get supportedList(): Features[] {
        const config = getConfig();
        return MODEL_SUPPORTED_FEATURES[config.model];
      },
      get isSummarizationSupported(): boolean {
        return this.supportedList.includes(Features.SUMMARIZATION);
      },
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

    workspace.onDidChangeConfiguration(
      (event) => {
        if (event.affectsConfiguration(CONFIG_KEY)) {
          this._emitCurrentState();
        }
      },
      null,
      extensionContext.subscriptions
    );
  }

  private _emitCurrentState(): void {
    this._emitter.emit(ExtensionState._stateChangedEvent, this._state);
  }

  set<K extends keyof IExtensionState>(key: K, value: IExtensionState[K]): void {
    this._state[key] = value;
    this._emitCurrentState();
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
