import { SidePanelMessageTypes } from '@shared/side-panel-message';
import { vscode } from '../../../utils/vscode';
import './CompletionSection.css';
import { VscodeIcon } from '../../shared/VscodeIcon/VscodeIcon';

interface CompletionSectionProps {
  isLoading: boolean;
}

export function CompletionSection({ isLoading }: CompletionSectionProps = { isLoading: false }): JSX.Element {
  const handleGenerateClick = () => {
    vscode.postMessage({
      type: SidePanelMessageTypes.GENERATE_COMPLETION_CLICK,
    });
  };

  return (
    <section className="completion-section">
      <h3>Code Completion</h3>
      <span>
        {/* TODO Detect OS and show specific keybindings */}
        {/* TODO Consider getting keybinding from package.json */}
        To generate inline code completion use combination <kbd>Ctrl+Alt+Space</kbd> or press the button below.
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
