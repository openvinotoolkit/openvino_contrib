import { SidePanelMessageTypes } from '@shared/side-panel-message';
import { vscode } from '../../../utils/vscode';
import { IExtensionState } from '@shared/extension-state';
import { ServerStatus as ServerStatusEnum } from '@shared/server-state';
import { StartingStages } from './StartingStages/StartingStages';
import { ServerStatus } from './ServerStatus/ServerStatus';
import './ServerSection.css';
import { ModelSelect } from './ModelSelect/ModelSelect';
import { ModelName } from '@shared/model';
import { DeviceSelect } from './DeviceSelect/DeviceSelect';
import { DeviceName } from '@shared/device';

interface ServerSectionProps {
  state: IExtensionState | null;
}

export function ServerSection({ state }: ServerSectionProps): JSX.Element {
  const handleStartServerClick = () => {
    vscode.postMessage({
      type: SidePanelMessageTypes.START_SERVER_CLICK,
    });
  };

  const handleStopServerClick = () => {
    vscode.postMessage({
      type: SidePanelMessageTypes.STOP_SERVER_CLICK,
    });
  };

  const handleShowServerLogClick = () => {
    vscode.postMessage({
      type: SidePanelMessageTypes.SHOW_SERVER_LOG_CLICK,
    });
  };

  const handleCheckConnectionClick = () => {
    vscode.postMessage({
      type: SidePanelMessageTypes.CHECK_CONNECTION_CLICK,
    });
  };

  const handleModelChange = (modelName: ModelName) => {
    vscode.postMessage({
      type: SidePanelMessageTypes.MODEL_CHANGE,
      payload: {
        modelName,
      },
    });
  };

  const handleDeviceChange = (deviceName: DeviceName) => {
    vscode.postMessage({
      type: SidePanelMessageTypes.DEVICE_CHANGE,
      payload: {
        deviceName,
      },
    });
  };

  if (!state) {
    return <>Extension state is not available</>;
  }

  const isServerStopped = state.server.status === ServerStatusEnum.STOPPED;
  const isServerStarting = state.server.status === ServerStatusEnum.STARTING;

  return (
    <section className="server-section">
      <h3>OpenVINO Code Server</h3>
      <ServerStatus status={state.server.status} connectionStatus={state.connectionStatus}></ServerStatus>
      <ModelSelect
        disabled={!isServerStopped}
        onChange={handleModelChange}
        selectedModelName={state.config.model}
        supportedFeatures={state.features.supportedList}
        serverStatus={state.server.status}
      ></ModelSelect>
      <DeviceSelect
        disabled={!isServerStopped}
        onChange={handleDeviceChange}
        selectedDeviceName={state.config.device}
        supportedFeatures={state.features.supportedList}
        serverStatus={state.server.status}
      ></DeviceSelect>
      {isServerStarting && <StartingStages currentStage={state.server.stage}></StartingStages>}
      <div className="button-group">
        {isServerStopped && <button onClick={handleStartServerClick}>Start Server</button>}
        {!isServerStopped && (
          <button className="secondary" onClick={handleStopServerClick}>
            Stop Server
          </button>
        )}
      </div>
      <div className="button-group">
        <a href="#" onClick={handleShowServerLogClick}>
          Show Server Log
        </a>
        <a href="#" onClick={handleCheckConnectionClick}>
          Check Connection
        </a>
      </div>
    </section>
  );
}
