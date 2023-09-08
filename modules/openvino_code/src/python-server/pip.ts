import { execCommand } from './commands-runner';
import type { PythonServerConfiguration } from './python-server-runner';
import { getVenvActivateCommand } from './virtual-environment';

export async function upgradePip(config: PythonServerConfiguration) {
  const { python, serverDir, proxyEnv, abortSignal, logger } = config;
  logger.info('Upgrading pip version...');

  const activateCommand = getVenvActivateCommand(config);
  const upgradeCommand = `${python} -m pip install --upgrade pip`;
  const command = `${activateCommand} && ${upgradeCommand}`;
  await execCommand(command, {
    logger,
    cwd: serverDir,
    abortSignal,
    env: { ...proxyEnv },
  });

  logger.info('Pip version upgraded');
}

export async function installRequirements(config: PythonServerConfiguration) {
  const { serverDir, proxyEnv, abortSignal, logger } = config;
  logger.info('Installing python requirements...');

  const activateCommand = getVenvActivateCommand(config);
  const installCommand = `pip install .`;
  const command = `${activateCommand} && ${installCommand}`;
  await execCommand(command, {
    logger,
    cwd: serverDir,
    abortSignal,
    env: { ...proxyEnv },
  });

  logger.info('Python requirements installed');
}
