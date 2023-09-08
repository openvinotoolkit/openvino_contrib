import { LogOutputChannel } from 'vscode';

enum LogLevel {
  TRACE = 'TRACE',
  DEBUG = 'DEBUG',
  INFO = 'INFO',
  WARNING = 'WARNING',
  ERROR = 'ERROR',
  CRITICAL = 'CRITICAL',
}

const serverLogPrefix = '[OpenVINO Code Server Log] ';
const timestampPattern = '\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2},\\d{3}';
const logLevelPattern = Object.values(LogLevel).join('|');

const serverLogRegExp = new RegExp(`^${timestampPattern} (?<level>${logLevelPattern}) (?<message>(?:.|\\n|\\r)+)`);

interface IServerLog {
  level: LogLevel;
  message: string;
}

const parseServerLogs = (output: string): (IServerLog | null)[] => {
  const serverLogs = output.split(serverLogPrefix).filter(Boolean);

  return serverLogs.map((log) => {
    const match = serverLogRegExp.exec(log);

    const level = match?.groups?.level as LogLevel | undefined;
    const message = match?.groups?.message;

    if (!level || !message) {
      return null;
    }

    return {
      level,
      message,
    };
  });
};

export const logServerMessage = (
  logger: LogOutputChannel,
  message: string,
  prefixer: (message: string) => string
): void => {
  const logLevelToMethodMap: Record<LogLevel, LogOutputChannel['trace' | 'debug' | 'info' | 'warn' | 'error']> = {
    [LogLevel.TRACE]: logger.trace.bind(logger),
    [LogLevel.DEBUG]: logger.debug.bind(logger),
    [LogLevel.INFO]: logger.info.bind(logger),
    [LogLevel.WARNING]: logger.warn.bind(logger),
    [LogLevel.ERROR]: logger.error.bind(logger),
    [LogLevel.CRITICAL]: logger.error.bind(logger),
  };

  const serverLogs = parseServerLogs(message);

  for (const serverLog of serverLogs) {
    if (!serverLog) {
      logger.debug(message);
    } else {
      const loggerMethod = logLevelToMethodMap[serverLog.level];

      loggerMethod(prefixer(serverLog.message));
    }
  }
};
