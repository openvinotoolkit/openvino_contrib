import { SidePanelMessageTypes } from '@shared/side-panel-message';
import { vscode } from '../../../utils/vscode';
import { IExtensionState } from '@shared/extension-state';
import { ServerStatus as ServerStatusEnum } from '@shared/server-state';
import { StartingStages } from './StartingStages/StartingStages';
import { ServerStatus } from './ServerStatus/ServerStatus';
import './ServerSection.css';

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

  const isServerStopped = state?.server.status === ServerStatusEnum.STOPPED;
  const isServerStarting = state?.server.status === ServerStatusEnum.STARTING;

  return (
    <section className="server-section">
      <h3>OpenVINO Code Server</h3>
      {state && <ServerStatus status={state.server.status} connectionStatus={state.connectionStatus}></ServerStatus>}
      {isServerStarting && state && <StartingStages currentStage={state.server.stage}></StartingStages>}
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
