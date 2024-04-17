import { ModelName } from '@shared/model';
import { Select, SelectOptionProps } from '../../../shared/Select/Select';
import { ServerStatus } from '@shared/server-state';
import { Features } from '@shared/features';

const options: SelectOptionProps<ModelName>[] = [
  { value: ModelName.CODE_T5_220M },
  { value: ModelName.DECICODER_1B_OPENVINO_INT8 },
  { value: ModelName.STABLECODE_COMPLETION_ALPHA_3B_4K_OPENVINO_INT8 },
  { value: ModelName.DEEPSEEK_CODER_1_3B },
  { value: ModelName.PHI_2_2_7B },
];

interface ModelSelectProps {
  disabled: boolean;
  selectedModelName: ModelName;
  onChange: (modelName: ModelName) => void;
  supportedFeatures: Features[];
  serverStatus: ServerStatus;
}

export const ModelSelect = ({
  disabled,
  selectedModelName,
  onChange,
  supportedFeatures,
  serverStatus,
}: ModelSelectProps): JSX.Element => {
  const isServerStopped = serverStatus === ServerStatus.STOPPED;
  return (
    <>
      <Select
        label="Model"
        options={options}
        selectedValue={selectedModelName}
        disabled={disabled}
        onChange={(value) => onChange(value)}
      ></Select>
      {isServerStopped && <span>Supported Features: {supportedFeatures.join(', ')}</span>}
    </>
  );
};
