import { EventEmitter as NodeEventEmitter } from 'stream';
import {
  InlineCompletionItem,
  InlineCompletionItemProvider,
  Position,
  Range,
  TextDocument,
  commands,
  window,
} from 'vscode';
import { EXTENSION_DISPLAY_NAME } from '../constants';
import completionService from './completion.service';

interface ICommandInlineCompletionItemProvider extends InlineCompletionItemProvider {
  triggerCompletion(onceCompleted: () => void): void;
}

/**
 * Trigger {@link ICommandInlineCompletionItemProvider.provideInlineCompletionItems}.
 * Executing editor.action.inlineSuggest.trigger command doesn't trigger inline completion when inlineSuggestionVisible context key is set.
 * Executing editor.action.inlineSuggest.hide before editor.action.inlineSuggest.trigger will make inline completion text bliks.
 * Replacing previous character before trigger seems to do the job.
 */
async function triggerInlineCompletionProvider(): Promise<void> {
  const editor = window.activeTextEditor;
  if (!editor) {
    return;
  }

  const document = editor.document;
  const activePosition = editor.selection.active;
  const activeOffset = document.offsetAt(activePosition);

  if (activeOffset === 0) {
    return;
  }

  const prevCharPosition = document.positionAt(activeOffset - 1);
  const replaceRange = new Range(prevCharPosition, activePosition);
  const value = document.getText(replaceRange);

  await editor.edit((edit) => edit.replace(replaceRange, value));
  await commands.executeCommand('editor.action.inlineSuggest.trigger');
}

export class StreamingCommandInlineCompletionItemProvider implements ICommandInlineCompletionItemProvider {
  private _isCommandRunning = false;

  private readonly _emitter = new NodeEventEmitter().setMaxListeners(1);

  private _streamBuffer: string = '';

  private readonly _commandCompletedEvent = 'CommandInlineCompletionItemProvider:completed';

  private _abortController = new AbortController();

  private _beforeComplete(): void {
    this._isCommandRunning = false;
    this._streamBuffer = '';
    this._abortController.abort();
    this._abortController = new AbortController();
    this._emitter.emit(this._commandCompletedEvent);
  }

  async triggerCompletion(onceCompleted: () => void) {
    this._emitter.once(this._commandCompletedEvent, onceCompleted);

    if (!window.activeTextEditor) {
      void window.showInformationMessage(`Please open a file first to use ${EXTENSION_DISPLAY_NAME}.`);
      this._beforeComplete();
      return;
    }

    if (this._isCommandRunning) {
      return;
    }

    this._isCommandRunning = true;

    void commands.executeCommand('workbench.action.focusStatusBar');
    void window.showTextDocument(window.activeTextEditor.document);

    await completionService.getCompletionStream(
      window.activeTextEditor.document,
      window.activeTextEditor.selection.active,
      async (chunk) => {
        this._streamBuffer += chunk;
        await triggerInlineCompletionProvider();
      },
      this._abortController.signal
    );

    this._isCommandRunning = false;
    await triggerInlineCompletionProvider();
  }

  stopGeneration() {
    this._abortController.abort();
  }

  cancelGeneration() {
    this._beforeComplete();
  }

  provideInlineCompletionItems(document: TextDocument, position: Position) {
    const buffer = this._streamBuffer;
    if (!this._isCommandRunning) {
      this._beforeComplete();
    }

    return [new InlineCompletionItem(`${buffer}`, new Range(position, position.translate(0, 1)))];
  }
}
