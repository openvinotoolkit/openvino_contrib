import { ServerStatus } from '@shared/server-state';
import { CompletionSection } from './components/sections/CompletionSection/CompletionSection';
import { OverviewSection } from './components/sections/OverviewSection/OverviewSection';
import { ServerSection } from './components/sections/ServerSection/ServerSection';
import { SummarizationSection } from './components/sections/SummarizationSection/SummarizationSection';
import { useExtensionState } from './use-extension-state';
import { vscode } from './utils/vscode';
import { SidePanelMessageTypes } from '@shared/side-panel-message';
import { ActionsSection } from './components/sections/ActionsSection/ActionsSection';

const initApp = (): void => {
  void vscode.postMessage({ type: SidePanelMessageTypes.GET_EXTENSION_STATE });
};

initApp();

function App(): JSX.Element {
  const [state] = useExtensionState();

  const isServerStopped = state?.server.status === ServerStatus.STOPPED;
  const isServerStarted = state?.server.status === ServerStatus.STARTED;

  return (
    <>
      <h2>OpenVINO AI Code Completion</h2>
      {isServerStopped && <OverviewSection></OverviewSection>}
      <ServerSection state={state}></ServerSection>
      {isServerStarted && (
        <>
          <CompletionSection isLoading={state?.isLoading}></CompletionSection>
          <SummarizationSection></SummarizationSection>
        </>
      )}
      <ActionsSection></ActionsSection>
    </>
  );
}

export default App;
