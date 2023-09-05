import { execCommand } from './commands-runner';
import type { PythonServerConfiguration } from './python-server-runner';
import { getVenvActivateCommand } from './virtual-environment';

// fixme: 'pip install --upgrade pip' doesn't fail on bad proxy settings. The command 'pip install --upgrade pip' exit with code 0 even there is a ProxyError in stderr.
// The plain 'pip install .' doesn't have such issue.
// fixme: 'pip install --upgrade pip' doesn't exit on stop signal. The plain 'pip install .' doesn't have such issue.
export async function upgradePip(config: PythonServerConfiguration) {
  const { python, serverDir, proxyEnv, stopSignal, logger } = config;
  logger.info('Upgrading pip version...');

  const activateCommand = getVenvActivateCommand(config);
  const upgradeCommand = `${python} -m pip install --upgrade pip`;
  const command = `${activateCommand} && ${upgradeCommand}`;
  await execCommand(command, {
    logger,
    cwd: serverDir,
    stopSignal,
    env: { ...proxyEnv },
  });

  logger.info('Pip version upgraded');
}

export async function installRequirements(config: PythonServerConfiguration) {
  const { serverDir, proxyEnv, stopSignal, logger } = config;
  logger.info('Installing python requirements...');

  const activateCommand = getVenvActivateCommand(config);
  const installCommand = `pip install .`;
  const command = `${activateCommand} && ${installCommand}`;
  await execCommand(command, {
    logger,
    cwd: serverDir,
    stopSignal,
    env: { ...proxyEnv },
  });

  logger.info('Python requirements installed');
}
