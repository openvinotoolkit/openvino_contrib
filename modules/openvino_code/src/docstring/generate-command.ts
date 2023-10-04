import { window } from 'vscode';
import { AutoDocstring } from './generate-docstring';
import { extensionState } from '../state';

export function generateCommandHandler() {
  const { isSummarizationSupported } = extensionState.state.features;
  const editor = window.activeTextEditor;
  if (!isSummarizationSupported || !editor) {
    return;
  }

  try {
    const autoDocstring = new AutoDocstring(editor);
    return autoDocstring.generate();
  } catch (error) {
    // todo: add proper logger
    console.error(error);
  }
}
