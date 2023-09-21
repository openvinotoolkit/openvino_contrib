import { platform } from 'os';

export enum OS {
  MAC_OS = 'darwin',
  LINUX = 'linux',
  WINDOWS = 'win32',
}

export function detectOs(): OS {
  const detectedPlatform = platform();
  if (detectedPlatform === OS.WINDOWS) {
    return OS.WINDOWS;
  }
  if (detectedPlatform === OS.LINUX) {
    return OS.LINUX;
  }
  if (detectedPlatform === OS.MAC_OS) {
    return OS.MAC_OS;
  }
  // as a fallback try linux
  return OS.LINUX;
}
