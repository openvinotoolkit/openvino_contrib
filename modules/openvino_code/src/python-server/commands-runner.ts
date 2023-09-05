import { ChildProcess, spawn, exec } from 'child_process';
import { StopSignal } from './stop-controller';
import { LogOutputChannel } from 'vscode';

export interface RunCommandOptions {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  stopSignal?: StopSignal;
  logger: LogOutputChannel;
  listeners?: Listeners;
}

interface Listeners {
  stdout: (data: string) => void;
}

const pidMessagePrefixer = (pid?: number) => (message: string) => `[Process: ${pid}] ${message}`;

export function spawnCommand(command: string, args: string[], options: RunCommandOptions) {
  const { env, cwd, stopSignal, logger, listeners } = options;

  if (stopSignal?.stopped) {
    logger.debug(`running command: ${command} ${args ? args?.join(' ') : ''} is skipped. Received stop signal`);
    return;
  }

  logger.debug(`running command: ${command} ${args ? args?.join(' ') : ''}`);

  const process = spawn(command, args, {
    cwd,
    env,
  });

  return waitForChildProcess(process, logger, stopSignal, listeners);
}

export function execCommand(command: string, options: RunCommandOptions) {
  const { env, cwd, stopSignal, logger, listeners } = options;

  if (stopSignal?.stopped) {
    logger.debug(`running command: ${command} is skipped. Received stop signal`);
    return;
  }

  logger.debug(`running command: ${command}`);

  const process = exec(command, {
    cwd,
    env,
  });

  return waitForChildProcess(process, logger, stopSignal, listeners);
}

async function waitForChildProcess(
  process: ChildProcess,
  logger: LogOutputChannel,
  stopSignal?: StopSignal,
  listeners?: Listeners
) {
  let result: string = '';
  let error: Error | null = null;
  const prefixMessage = pidMessagePrefixer(process.pid);

  const stopSignalHandler = () => {
    logger.debug(prefixMessage('killing process'));
    process.kill();
  };
  stopSignal?.once(stopSignalHandler);

  return new Promise<string | null>((resolve, reject) => {
    process.stdout?.on('data', (data) => {
      const textData = String(data).trim();

      logger.debug(prefixMessage(textData));

      if (listeners?.stdout) {
        listeners?.stdout(textData);
      } else {
        // do not accumulate logs if stdout listener passed
        result += data;
      }
    });

    process.stderr?.on('data', (data) => {
      const textData = String(data).trim();
      logger.error(prefixMessage(textData));
      if (!error) {
        error = new Error(textData);
      }
    });

    process.on('error', (err) => {
      logger.error(prefixMessage(err.message));
      error = err;
    });

    process.on('close', (code, signal) => {
      logger.debug(prefixMessage(`exited with code: ${code} and signal: ${signal}`));
      stopSignal?.removeListener(stopSignalHandler);

      if (code === 0) {
        resolve(result);
      } else {
        reject(error);
      }
    });
  });
}
