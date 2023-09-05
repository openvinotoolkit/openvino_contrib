import { LogOutputChannel } from 'vscode';
import { execCommand } from './commands-runner';

export type PythonExecutable = 'python' | 'python3';
export type Version = [major: string | number, minor: string | number];

export async function getPythonExecutable(
  requriedVersion: Version,
  logger: LogOutputChannel
): Promise<PythonExecutable> {
  logger.info('Finding Python executable...');

  const executables: [PythonExecutable, PythonExecutable] = ['python', 'python3'];
  let result: PythonExecutable | null = null;
  const errors: unknown[] = [];

  for (const executable of executables) {
    try {
      await verifyPythonVersion(executable, requriedVersion, logger);
      result = executable;
    } catch (e) {
      errors.push(e);
    }
  }

  if (result) {
    logger.info(`Python executable: ${result}`);
    return result;
  }

  const errorMessage = ['Cannot find python executable.'];
  if (errors.length) {
    errorMessage.push(' Next error(s) occured:\n');
    errorMessage.push(`${String(errors[0])}\n`);
  }

  if (errors[1]) {
    errorMessage.push(`${String(errors[1])}\n`);
  }

  throw new Error(errorMessage.join(''));
}

async function verifyPythonVersion(
  executable: PythonExecutable,
  requriedVersion: Version,
  logger: LogOutputChannel
): Promise<void> {
  const command = `${executable} --version`;
  const commandResult = await execCommand(command, {
    logger,
  });

  if (!commandResult) {
    throw new Error(`Cannot execute command: ${command}`);
  }

  const versionRegex = /[\d.]+/;
  const match = commandResult.match(versionRegex);
  if (!match) {
    throw new Error(`Cannot find python version`);
  }

  const [major, minor] = match[0].split('.');
  if (!isVersionCorrect([major, minor], requriedVersion)) {
    throw new Error(`Required python version: ${requriedVersion.join('.')}. Actual: ${match[0]}`);
  }
}

function isVersionCorrect(actual: Version, required: Version): boolean {
  return Number(actual[0]) >= Number(required[0]) && Number(actual[1]) >= Number(required[1]);
}
