import { ServerStatus as ServerStausEnum } from '@shared/server-state';
import { VscodeIcon } from '../../../shared/VscodeIcon/VscodeIcon';
import { ConnectionStatus } from '@shared/extension-state';
import './ServerStatus.css';

const serverStatusLabelsMap = {
  [ServerStausEnum.STOPPED]: 'Stopped',
  [ServerStausEnum.STARTING]: 'Starting',
  [ServerStausEnum.STARTED]: 'Running',
};

const serverStatusIconsMap = {
  [ServerStausEnum.STOPPED]: 'circle-slash',
  [ServerStausEnum.STARTING]: 'loading',
  [ServerStausEnum.STARTED]: 'vm-running',
};

const connectionStatusLabelsMap = {
  [ConnectionStatus.NOT_AVAILABLE]: 'Not Connected',
  [ConnectionStatus.PENDING]: 'Pending',
  [ConnectionStatus.AVAILABLE]: 'Connected',
};

const connectionStatusIconsMap = {
  [ConnectionStatus.NOT_AVAILABLE]: 'debug-disconnect',
  [ConnectionStatus.PENDING]: 'loading',
  [ConnectionStatus.AVAILABLE]: 'pass',
};

interface ServerStatusProps {
  status: ServerStausEnum;
  connectionStatus: ConnectionStatus;
}

export const ServerStatus = ({ status, connectionStatus }: ServerStatusProps): JSX.Element => (
  <div className="server-status">
    <span className="server-status-item">
      <VscodeIcon iconName={serverStatusIconsMap[status]} spin={status === ServerStausEnum.STARTING}></VscodeIcon>
      &nbsp;
      {serverStatusLabelsMap[status]}
    </span>
    <span className="server-status-item">
      <VscodeIcon
        iconName={connectionStatusIconsMap[connectionStatus]}
        spin={connectionStatus === ConnectionStatus.PENDING}
      ></VscodeIcon>
      &nbsp;
      {connectionStatusLabelsMap[connectionStatus]}
    </span>
  </div>
);
