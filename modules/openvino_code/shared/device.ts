import { Features } from './features';

enum DeviceId {
  CPU = 'CPU',
  GPU = 'GPU',
  NPU = 'NPU',
  }

export enum DeviceName {
  CPU = 'CPU',
  GPU = 'GPU',
  NPU = 'NPU',
  } 

export const DEVICE_NAME_TO_ID_MAP: Record<DeviceName, DeviceId> = {
  [DeviceName.CPU]: DeviceId.CPU,
  [DeviceName.GPU]: DeviceId.GPU,
  [DeviceName.NPU]: DeviceId.NPU,
};

export const DEVICE_SUPPORTED_FEATURES: Record<DeviceName, Features[]> = {
  [DeviceName.CPU]: [Features.CODE_COMPLETION, Features.SUMMARIZATION, Features.FIM],
  [DeviceName.GPU]: [Features.CODE_COMPLETION, Features.SUMMARIZATION, Features.FIM],
  [DeviceName.NPU]: [Features.CODE_COMPLETION, Features.SUMMARIZATION, Features.FIM],
};
