import { join } from 'path';
import { execCommand } from './commands-runner';
import { OS } from './detect-os';
import { PythonExecutable } from './detect-python';
import type { PythonServerConfiguration } from './python-server-runner';
import { LogOutputChannel } from 'vscode';
import { stat } from 'fs/promises';

export async function createVenv(
  python: PythonExecutable,
  directory: string,
  venvName: string,
  logger: LogOutputChannel
) {
  logger.info('Creating virtual environment...');

  const command = `${python} -m venv ${venvName}`;
  await execCommand(command, {
    cwd: directory,
    logger,
  });

  logger.info('Virtual environment created');
}

export async function checkActivatedVenv(config: PythonServerConfiguration) {
  const activateCommand = getVenvActivateCommand(config);
  await execCommand(activateCommand, {
    cwd: config.serverDir,
    logger: config.logger,
  });
}

export function getVenvActivateCommand({ os, venvDirName, serverDir }: PythonServerConfiguration) {
  if (os === OS.WINDOWS) {
    return join(serverDir, venvDirName, 'Scripts', 'activate');
  }
  const venvPath = join(serverDir, venvDirName, 'bin', 'activate');
  return `. ${venvPath}`;
}

export async function getVenvPythonPath({ python, os, venvDirName, serverDir, logger }: PythonServerConfiguration) {
  /**
   * On windows machine it is possible that in venv could be only 'python' executable even for creation 'python3' was used
   */

  const binariesDir = join(serverDir, venvDirName, os === OS.WINDOWS ? 'Scripts' : 'bin');
  const extension = os === OS.WINDOWS ? '.exe' : '';

  try {
    const venvPythonPath = join(binariesDir, `${python}${extension}`);
    await stat(venvPythonPath);
    return venvPythonPath;
  } catch (e) {
    logger.debug(`python executable with system matching name ${python} not found in venv`);
  }

  // get other 'python' name
  const candidates = new Set(['python', 'python3']);
  candidates.delete(python);
  const candidate = Array.from(candidates.values())[0];

  logger.debug(`Checking "${candidate}" in venv`);
  const candidateVenvPath = join(binariesDir, `${candidate}${extension}`);
  await stat(candidateVenvPath);

  logger.debug(`"${candidate}" found in venv`);
  return candidateVenvPath;
}

export function getVenvPip({ os, venvDirName, serverDir }: PythonServerConfiguration) {
  if (os === OS.WINDOWS) {
    return join(serverDir, venvDirName, 'Scripts', 'pip');
  }
  return join(serverDir, venvDirName, 'bin', 'pip');
}
