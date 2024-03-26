//import { ModelName } from '@shared/model';
import { DeviceName } from '@shared/device';
import { Select, SelectOptionProps } from '../../../shared/Select/Select';
import { ServerStatus } from '@shared/server-state';
import { Features } from '@shared/features';

const options: SelectOptionProps<DeviceName>[] = [
  { value: DeviceName.CPU },
  { value: DeviceName.GPU },
  { value: DeviceName.NPU },
];

interface DeviceSelectProps {
  disabled: boolean;
  selectedDeviceName: DeviceName;
  onChange: (deviceName: DeviceName) => void;
  supportedFeatures: Features[];
  serverStatus: ServerStatus;
}

export const DeviceSelect = ({
  disabled,
  selectedDeviceName,
  onChange,
  supportedFeatures,
  serverStatus,
}: DeviceSelectProps): JSX.Element => {
  const isServerStopped = serverStatus === ServerStatus.STOPPED;
  return (
    <>
      <Select
        label="Device"
        options={options}
        selectedValue={selectedDeviceName}
        disabled={disabled}
        onChange={(value) => onChange(value)}
      ></Select>
      {isServerStopped && <span>Supported Features: {supportedFeatures.join(', ')}</span>}
    </>
  );
};
