import { workspace } from 'vscode';

export interface ProxyEnv {
  http_proxy?: string;
  https_proxy?: string;
  no_proxy?: string;
  PIP_PROXY?: string;
}

export function getProxyEnv(): ProxyEnv | undefined {
  return getVSCodeProxyEnv() || getSystemProxyEnv();
}

function getVSCodeProxyEnv(): ProxyEnv | undefined {
  const httpProxy = workspace.getConfiguration().get<string>('http.proxy');

  if (!httpProxy) {
    return;
  }

  return {
    http_proxy: httpProxy,
    https_proxy: httpProxy,
    PIP_PROXY: httpProxy,
  };
}

function getSystemProxyEnv(): ProxyEnv | undefined {
  const settings: ProxyEnv = {
    http_proxy: getSystemEnv('http_proxy'),
    https_proxy: getSystemEnv('https_proxy'),
    no_proxy: getSystemEnv('no_proxy'),
    PIP_PROXY: getSystemEnv('http_proxy'),
  };

  const nonEmptySettings: ProxyEnv = {};
  for (const [name, value] of Object.entries(settings)) {
    if (value) {
      nonEmptySettings[name as keyof ProxyEnv] = value as string;
    }
  }
  if (!Object.keys(nonEmptySettings).length) {
    return;
  }
  return nonEmptySettings;
}

function getSystemEnv(name: string): string | undefined {
  return process.env[name.toLowerCase()] || process.env[name.toUpperCase()];
}
