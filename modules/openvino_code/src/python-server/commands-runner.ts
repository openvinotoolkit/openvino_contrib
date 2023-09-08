import { ChildProcess, spawn, exec } from 'child_process';
import { LogOutputChannel } from 'vscode';
import { logServerMessage } from './server-log';

export interface RunCommandOptions {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  abortSignal?: AbortSignal;
  logger: LogOutputChannel | null;
  listeners?: Listeners;
}

interface Listeners {
  stdout: (data: string) => void;
}

export const isAbortError = (error: Error): boolean => error.name === 'AbortError';

const pidMessagePrefixer = (pid?: number) => (message: string) => `[Process: ${pid}] ${message}`;

export function spawnCommand(command: string, args: string[], options: RunCommandOptions) {
  const { env, cwd, abortSignal, logger, listeners } = options;

  if (abortSignal?.aborted) {
    logger?.debug(`running command: ${command} ${args ? args?.join(' ') : ''} is skipped. Received stop signal`);
    return;
  }

  logger?.debug(`running command: ${command} ${args ? args?.join(' ') : ''}`);

  const process = spawn(command, args, {
    cwd,
    env,
    signal: abortSignal,
  });

  return waitForChildProcess(process, logger, abortSignal, listeners);
}

export function execCommand(command: string, options: RunCommandOptions) {
  const { env, cwd, abortSignal, logger, listeners } = options;

  if (abortSignal?.aborted) {
    logger?.debug(`running command: ${command} is skipped. Received stop signal`);
    return;
  }

  logger?.debug(`running command: ${command}`);

  const process = exec(command, {
    cwd,
    env,
    signal: abortSignal,
  });

  return waitForChildProcess(process, logger, abortSignal, listeners);
}

async function waitForChildProcess(
  process: ChildProcess,
  logger: RunCommandOptions['logger'],
  abortSignal?: AbortSignal,
  listeners?: Listeners
) {
  let result: string = '';
  let error: Error | null = null;
  const prefixMessage = pidMessagePrefixer(process.pid);

  const stopSignalHandler = () => {
    logger?.debug(prefixMessage('killing process'));
    // TODO Consider removing explicit process kill and rely on AbortSignal only
    process.kill();
  };

  abortSignal?.addEventListener('abort', stopSignalHandler, { once: true });

  return new Promise<string | null>((resolve, reject) => {
    process.stdout?.on('data', (data) => {
      const textData = String(data).trim();

      if (logger) {
        logServerMessage(logger, textData, prefixMessage);
      }

      if (listeners?.stdout) {
        listeners?.stdout(textData);
      } else {
        // do not accumulate logs if stdout listener passed
        result += data;
      }
    });

    process.stderr?.on('data', (data) => {
      const textData = String(data).trim();
      logger?.error(prefixMessage(textData));
      if (!error) {
        error = new Error(textData);
      }
    });

    process.on('error', (err) => {
      if (isAbortError(err)) {
        logger?.debug(prefixMessage(err.message));
      } else {
        logger?.error(prefixMessage(err.message));
      }
      error = err;
    });

    process.on('close', (code, signal) => {
      logger?.debug(prefixMessage(`exited with code: ${code} and signal: ${signal}`));

      abortSignal?.removeEventListener('abort', stopSignalHandler);

      if (code === 0) {
        resolve(result);
      } else {
        reject(error);
      }
    });
  });
}
