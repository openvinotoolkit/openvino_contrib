import { Features } from './features';

enum ModelId {
  CODE_T5_220M = 'Salesforce/codet5p-220m-py',
  DECICODER_1B_OPENVINO_INT8 = 'chgk13/decicoder-1b-openvino-int8',
}

export enum ModelName {
  CODE_T5_220M = 'codet5p-220m-py',
  DECICODER_1B_OPENVINO_INT8 = 'decicoder-1b-openvino-int8',
}

export const MODEL_NAME_TO_ID_MAP: Record<ModelName, ModelId> = {
  [ModelName.CODE_T5_220M]: ModelId.CODE_T5_220M,
  [ModelName.DECICODER_1B_OPENVINO_INT8]: ModelId.DECICODER_1B_OPENVINO_INT8,
};

export const MODEL_SUPPORTED_FEATURES: Record<ModelName, Features[]> = {
  [ModelName.CODE_T5_220M]: [Features.CODE_COMPLETION],
  [ModelName.DECICODER_1B_OPENVINO_INT8]: [Features.CODE_COMPLETION, Features.SUMMARIZATION],
};
