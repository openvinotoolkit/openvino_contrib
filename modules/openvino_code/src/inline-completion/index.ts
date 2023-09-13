import { IExtensionState } from '@shared/extension-state';
import { ExtensionContext } from 'vscode';
import { IExtensionComponent } from '../extension-component.interface';
import { extensionState } from '../state';
import { inlineCompletion as baseInlineCompletion } from './inline-completion-component';
import { streamingInlineCompletion } from './streaming-inline-completion-component';

class InlineCompletion implements IExtensionComponent {
  private _context: ExtensionContext | null = null;
  private _listener = ({ config }: IExtensionState) => this.activateCompletion(config.streamInlineCompletion);

  activate(context: ExtensionContext): void {
    this._context = context;
    this.activateCompletion(extensionState.config.streamInlineCompletion);
    extensionState.subscribe(this._listener);
  }

  deactivate(): void {
    streamingInlineCompletion.deactivate();
    baseInlineCompletion.deactivate();
    extensionState.unsubscribe(this._listener);
  }

  activateCompletion(streaming: boolean) {
    if (!this._context) {
      return;
    }
    baseInlineCompletion.deactivate();
    streamingInlineCompletion.deactivate();

    if (streaming) {
      streamingInlineCompletion.activate(this._context);
    } else {
      baseInlineCompletion.activate(this._context);
    }
  }
}

export const inlineCompletion = new InlineCompletion();
