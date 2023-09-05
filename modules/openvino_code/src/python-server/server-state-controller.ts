import { EventEmitter } from 'node:events';
import { ServerState, ServerStatus, ServerStartingStage, INITIAL_SERVER_STATE } from '@shared/server-state';
import { Disposable } from 'vscode';

const changeEventName = 'change';

// eslint-disable-next-line @typescript-eslint/no-unsafe-declaration-merging
declare interface ServerStateReporter {
  on(eventName: typeof changeEventName, listener: (serverState: ServerState) => void): this;
  off(eventName: typeof changeEventName, listener: (serverState: ServerState) => void): this;
}

// eslint-disable-next-line @typescript-eslint/no-unsafe-declaration-merging
class ServerStateReporter extends EventEmitter {
  constructor(private _controller: ServerStateController) {
    super();
  }

  get state() {
    return this._controller.state;
  }

  subscribeToStateChange(listener: (serverState: ServerState) => void): Disposable {
    this.on(changeEventName, listener);
    return new Disposable(() => this.off('change', listener));
  }
}

export class ServerStateController {
  private _state: ServerState = INITIAL_SERVER_STATE;

  get state() {
    return this._state;
  }

  readonly reporter = new ServerStateReporter(this);

  setStatus(status: ServerStatus) {
    this._state.status = status;
    this._emitChange();
  }

  setStage(stage: ServerStartingStage | null) {
    this._state.stage = stage;
    this._emitChange();
  }

  private _emitChange() {
    this.reporter.emit(changeEventName, this._state);
  }
}
