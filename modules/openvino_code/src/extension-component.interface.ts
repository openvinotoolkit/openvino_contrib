import { ExtensionContext } from 'vscode';

export interface IExtensionComponent {
  activate(context: ExtensionContext): void;
  deactivate(): void;
}
