import { SidePanelMessageTypes } from '@shared/side-panel-message';
import { vscode } from '../../../utils/vscode';
import './ActionsSection.css';

const hangleShowLogClick = () => {
  vscode.postMessage({
    type: SidePanelMessageTypes.SHOW_EXTENSION_LOG_CLICK,
  });
};

const hangleSettingsClick = () => {
  vscode.postMessage({
    type: SidePanelMessageTypes.SETTINGS_CLICK,
  });
};

export function ActionsSection(): JSX.Element {
  return (
    <section className="actions-section">
      <a className="show-log-link" href="#" onClick={hangleShowLogClick}>
        Show Extension Log
      </a>
      <a className="settings-link" href="#" onClick={hangleSettingsClick}>
        Extension Settings
      </a>
    </section>
  );
}
