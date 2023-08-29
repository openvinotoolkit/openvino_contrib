import { spawnCommand } from './commands-runner';
import { getVenvPythonPath } from './virtual-environment';
import { PythonExecutable, Version, getPythonExecutable } from './detect-python';
import { OS, detectOs } from './detect-os';
import { createVenv, checkActivatedVenv } from './virtual-environment';
import { upgradePip, installRequirements } from './pip';
import { StopSignal, StopController } from './stop-controller';
import { ProxyEnv, getProxyEnv } from './proxy';
import { ServerStateController } from './server-state-controller';
import { ServerStatus, ServerStartingStage } from '@shared/server-state';
import { EXTENSION_SERVER_DISPLAY_NAME } from '../constants';
import { LogOutputChannel, window } from 'vscode';
import { join } from 'path';

const SERVER_STARTED_STDOUT_ANCHOR = 'OpenVINO Code Server started';

interface ServerHooks {
  onStarted: () => void;
}

async function runServer(config: PythonServerConfiguration, hooks?: ServerHooks) {
  const { serverDir, proxyEnv, stopSignal, logger } = config;
  logger.info('Starting server...');

  const venvPython = await getVenvPythonPath(config);
  let started = false;

  function stdoutListener(data: string) {
    if (started) {
      return;
    }

    if (data.includes(SERVER_STARTED_STDOUT_ANCHOR)) {
      hooks?.onStarted();
      started = true;
    }
  }

  await spawnCommand(venvPython, ['main.py', '--model', 'chgk13/decicoder-1b-openvino-int8'], {
    logger,
    cwd: serverDir,
    stopSignal,
    env: { ...proxyEnv },
    listeners: { stdout: stdoutListener },
  });
}

const logger = window.createOutputChannel(EXTENSION_SERVER_DISPLAY_NAME, { log: true });

const SERVER_DIR = join(__dirname, 'server');

export interface PythonServerConfiguration {
  python: PythonExecutable;
  os: OS;
  venvDirName: string;
  serverDir: string;
  proxyEnv?: ProxyEnv;
  stopSignal: StopSignal;
  logger: LogOutputChannel;
}

export class NativePythonServerRunner {
  static readonly REQUIRED_PYTHON_VERSION: Version = [3, 8];
  static readonly VENV_DIR_NAME = '.venv';

  private _stopController = new StopController();

  private readonly _stateController = new ServerStateController();
  get stateReporter() {
    return this._stateController.reporter;
  }

  async start() {
    if (this._stateController.state.status === ServerStatus.STARTED) {
      logger.info('Server is already started. Skipping start command');
      return;
    }

    this._stateController.setStatus(ServerStatus.STARTING);

    try {
      logger.info('Strating Server using python vitual environment...');
      await this._start();
      logger.info('Server stopped');
    } catch (e) {
      const error = e instanceof Error ? e : String(e);
      logger.error(`Server stopped with error:`);
      logger.error(error);
    } finally {
      this._stateController.setStage(null);
      this._stateController.setStatus(ServerStatus.STOPPED);
    }
  }

  async _start() {
    const os = detectOs();
    logger.info(`System detected: ${os}`);

    this._stateController.setStage(ServerStartingStage.DETECT_SYSTEM_PYTHON);

    const python: PythonExecutable = await getPythonExecutable(
      NativePythonServerRunner.REQUIRED_PYTHON_VERSION,
      logger
    );

    this._stateController.setStage(ServerStartingStage.CREATE_VENV);

    await createVenv(python, SERVER_DIR, NativePythonServerRunner.VENV_DIR_NAME, logger);

    const proxyEnv = getProxyEnv();
    if (proxyEnv) {
      logger.info('Applying proxy settings:');
      logger.info(JSON.stringify(proxyEnv, null, 2));
    }

    const config: PythonServerConfiguration = {
      python,
      os,
      proxyEnv: getProxyEnv(),
      serverDir: SERVER_DIR,
      venvDirName: NativePythonServerRunner.VENV_DIR_NAME,
      stopSignal: this._stopController.signal,
      logger,
    };

    this._stateController.setStage(ServerStartingStage.CHECK_VENV_ACTIVATION);

    await checkActivatedVenv(config);

    this._stateController.setStage(ServerStartingStage.UPGRADE_PIP);

    await upgradePip(config);

    this._stateController.setStage(ServerStartingStage.INSTALL_REQUIREMENTS);

    await installRequirements(config);

    this._stateController.setStage(ServerStartingStage.START_SERVER);

    await runServer(config, {
      onStarted: () => {
        this._stateController.setStatus(ServerStatus.STARTED);
        this._stateController.setStage(null);
      },
    });
  }

  stop() {
    logger.info('Stopping...');
    this._stopController.stop();
    this._stopController = new StopController();
  }
}
