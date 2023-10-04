import { IExtensionState } from '@shared/extension-state';
import { useEffect, useState } from 'react';

export const useExtensionState = () => {
  const [state, setState] = useState<IExtensionState | null>(null);
  const eventType = 'message';
  useEffect(() => {
    const listener = ({ data }: MessageEvent<IExtensionState>) => setState(data);
    window.addEventListener(eventType, listener);
    return () => {
      window.removeEventListener(eventType, listener);
    };
  }, [eventType]);
  return [state];
};
