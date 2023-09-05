import { EventEmitter } from 'node:events';

interface Listener {
  (...args: unknown[]): unknown;
}

export class StopSignal {
  stopped = false;
  once: (fn: Listener) => void;
  removeListener: (fn: Listener) => void;

  constructor(once: (fn: Listener) => void, removeListener: (fn: Listener) => void) {
    this.once = once;
    this.removeListener = removeListener;
  }
}

export class StopController {
  readonly signal: StopSignal;

  private readonly _emmiter = new EventEmitter();
  private static _stopEventName = 'stop';

  constructor() {
    this.signal = new StopSignal(this.once.bind(this), this.removeListener.bind(this));
  }

  private once(fn: Listener) {
    this._emmiter.addListener(StopController._stopEventName, fn);
  }

  private removeListener(fn: Listener) {
    this._emmiter.removeListener(StopController._stopEventName, fn);
  }

  stop() {
    this._emmiter.emit(StopController._stopEventName);
    this.signal.stopped = true;
  }
}
