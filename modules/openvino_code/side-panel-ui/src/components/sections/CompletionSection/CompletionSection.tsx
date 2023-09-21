import { SidePanelMessageTypes } from '@shared/side-panel-message';
import { vscode } from '../../../utils/vscode';
import './CompletionSection.css';
import { VscodeIcon } from '../../shared/VscodeIcon/VscodeIcon';

interface CompletionSectionProps {
  isLoading: boolean;
  platform: NodeJS.Platform;
}

export function CompletionSection({ isLoading, platform }: CompletionSectionProps): JSX.Element {
  const handleGenerateClick = () => {
    vscode.postMessage({
      type: SidePanelMessageTypes.GENERATE_COMPLETION_CLICK,
    });
  };

  const platformKeyBinding = platform === 'darwin' ? 'Cmd+Alt+Space' : 'Ctrl+Alt+Space';

  return (
    <section className="completion-section">
      <h3>Code Completion</h3>
      <span>
        {/* TODO Consider getting keybinding from package.json */}
        To generate inline code completion use combination <kbd>{platformKeyBinding}</kbd> or press the button below.
      </span>
      <br />
      <button className="generate-button" onClick={handleGenerateClick} disabled={isLoading}>
        {isLoading && <VscodeIcon iconName="loading" spin></VscodeIcon>}
        <span>{isLoading ? 'Generating' : 'Generate'} Code Completion</span>
      </button>
      <br />
    </section>
  );
}
