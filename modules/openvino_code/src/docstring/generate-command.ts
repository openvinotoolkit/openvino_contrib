import { window } from "vscode";
import { AutoDocstring } from "./generate-docstring";

export function generateCommandHandler() {
  const editor = window.activeTextEditor;
  if (!editor) {
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
