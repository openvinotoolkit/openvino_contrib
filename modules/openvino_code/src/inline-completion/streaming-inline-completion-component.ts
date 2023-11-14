import {
  Disposable,
  ExtensionContext,
  Position,
  TextEditorSelectionChangeEvent,
  commands,
  languages,
  window,
} from 'vscode';
import { COMMANDS, EXTENSION_CONTEXT_STATE } from '../constants';
import { IExtensionComponent } from '../extension-component.interface';
import { extensionState } from '../state';
import { StreamingCommandInlineCompletionItemProvider } from './streaming-command-inline-completion-provider';
import { notificationService } from '../services/notification.service';

class StreamingInlineCompletion implements IExtensionComponent {
  private _disposables: Disposable[] = [];

  activate(context: ExtensionContext): void {
    // Register Inline Completion triggered by command
    const commandInlineCompletionProvider = new StreamingCommandInlineCompletionItemProvider();

    let generationDisposables: Disposable[] = [];

    function disposeGenerationDisposables() {
      generationDisposables?.forEach((disposable) => {
        disposable.dispose();
      });
      generationDisposables = [];
    }

    const generateCommandDisposable = commands.registerCommand(COMMANDS.GENERATE_INLINE_COPMLETION, () => {
      if (!extensionState.get('isServerAvailable')) {
        notificationService.showServerNotAvailableMessage(extensionState.state);
        return;
      }
      if (extensionState.get('isLoading') && window.activeTextEditor) {
        void window.showTextDocument(window.activeTextEditor.document);
        return;
      }

      extensionState.set('isLoading', true);

      disposeGenerationDisposables();

      const detectPositionChange = getPositionChangeDetector(
        window.activeTextEditor!.document.fileName,
        window.activeTextEditor!.selection.active
      );

      generationDisposables.push(
        commands.registerCommand(COMMANDS.STOP_GENERATION, () => {
          commandInlineCompletionProvider.stopGeneration();
        }),
        window.onDidChangeTextEditorSelection((change) => {
          if (!detectPositionChange(change)) {
            return;
          }
          commandInlineCompletionProvider.cancelGeneration();
        }),
        languages.registerInlineCompletionItemProvider({ pattern: '**' }, commandInlineCompletionProvider)
      );

      void commands.executeCommand('setContext', EXTENSION_CONTEXT_STATE.GENERATING, true);

      void commandInlineCompletionProvider.triggerCompletion(() => {
        disposeGenerationDisposables();
        extensionState.set('isLoading', false);
        // TODO: handle unsetting context on error thrown from triggerCompletion
        void commands.executeCommand('setContext', EXTENSION_CONTEXT_STATE.GENERATING, false);
      });
    });

    const acceptCommandDisposable = commands.registerCommand(COMMANDS.ACCEPT_INLINE_COMPLETION, () => {
      void commands.executeCommand('editor.action.inlineSuggest.commit');
    });

    context.subscriptions.push(generateCommandDisposable, acceptCommandDisposable);
    this._disposables.push(generateCommandDisposable, acceptCommandDisposable);
  }

  deactivate(): void {
    this._disposables.forEach((disposable) => {
      disposable.dispose();
    });
    this._disposables = [];
  }
}

export const streamingInlineCompletion = new StreamingInlineCompletion();

function getPositionChangeDetector(fileName: string, initialPosition: Position) {
  let prevPosition: Position = initialPosition;
  return (change: TextEditorSelectionChangeEvent) => {
    if (fileName !== change.textEditor.document.fileName) {
      return false;
    }

    const currentPosition = change.selections[0].active;

    if (prevPosition.isEqual(currentPosition)) {
      return false;
    }

    prevPosition = currentPosition;
    return true;
  };
}
