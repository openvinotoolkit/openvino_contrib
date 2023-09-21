import {
  InlineCompletionItem,
  InlineCompletionItemProvider,
  InlineCompletionList,
  Position,
  TextDocument,
  commands,
  window,
} from 'vscode';
import { EventEmitter } from 'stream';
import completionService from './completion.service';
import { EXTENSION_DISPLAY_NAME } from '../constants';

interface ICommandInlineCompletionItemProvider extends InlineCompletionItemProvider {
  triggerCompletion(onceCompleted: () => void): void;
}

export class CommandInlineCompletionItemProvider implements ICommandInlineCompletionItemProvider {
  private _isCommandRunning = false;

  private _completionItems: InlineCompletionItem[] | InlineCompletionList = [];

  private readonly _emitter = new EventEmitter().setMaxListeners(1);

  private readonly _commandCompletedEvent = 'CommandInlineCompletionItemProvider:completed';

  private _beforeComplete(): void {
    this._isCommandRunning = false;
    this._emitter.emit(this._commandCompletedEvent);
  }

  async triggerCompletion(onceCompleted: () => void): Promise<void> {
    this._emitter.once(this._commandCompletedEvent, onceCompleted);

    if (!window.activeTextEditor) {
      void window.showInformationMessage(`Please open a file first to use ${EXTENSION_DISPLAY_NAME}.`);
      this._beforeComplete();
      return;
    }

    this._isCommandRunning = true;

    await commands.executeCommand('workbench.action.focusStatusBar');
    await window.showTextDocument(window.activeTextEditor.document);
    await commands.executeCommand('editor.action.inlineSuggest.trigger');
  }

  async provideInlineCompletionItems(document: TextDocument, position: Position) {
    if (!this._isCommandRunning) {
      this._beforeComplete();
      return this._completionItems;
    }

    const completionItems = await completionService.getCompletion(document, position);

    if (!completionItems?.length) {
      this._beforeComplete();
      return [];
    }

    this._completionItems = completionItems;
    this._beforeComplete();
    return this._completionItems;
  }
}
