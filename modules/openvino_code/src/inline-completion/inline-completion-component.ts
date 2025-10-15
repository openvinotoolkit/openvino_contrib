import { Disposable, ExtensionContext, commands, window, languages } from 'vscode';
import { COMMANDS } from '../constants';
import { IExtensionComponent } from '../extension-component.interface';
import { notificationService } from '../services/notification.service';
import { extensionState } from '../state';
import { CommandInlineCompletionItemProvider } from './command-inline-completion-provider';
import { setIsGeneralTabActive } from './tab';

class InlineCompletion implements IExtensionComponent {
  private _disposables: Disposable[] = [];

  activate(context: ExtensionContext): void {
    // Register Inline Completion triggered by command
    const commandInlineCompletionProvider = new CommandInlineCompletionItemProvider();

    let commandInlineCompletionDisposable: Disposable;

    function generateFunction(): void {
        if (!extensionState.get('isServerAvailable')) {
            notificationService.showServerNotAvailableMessage(extensionState.state);
            return;
        }
        if (extensionState.get('isLoading') && window.activeTextEditor) {
            void window.showTextDocument(window.activeTextEditor.document);
            return;
        }
    
        extensionState.set('isLoading', true);
    
        if (commandInlineCompletionDisposable) {
            commandInlineCompletionDisposable.dispose();
        }
    
        commandInlineCompletionDisposable = languages.registerInlineCompletionItemProvider(
            { pattern: '**' },
            commandInlineCompletionProvider
        );
    
        void commandInlineCompletionProvider.triggerCompletion(() => {
            commandInlineCompletionDisposable.dispose();
            extensionState.set('isLoading', false);
        });
    }

    const acceptCommandDisposable = commands.registerCommand(COMMANDS.ACCEPT_INLINE_COMPLETION, () => {
      void commands.executeCommand('editor.action.inlineSuggest.commit');
    });

    const generateCommandDisposable = commands.registerCommand(COMMANDS.GENERATE_INLINE_COPMLETION, () => {
      setIsGeneralTabActive(false);
      generateFunction();
    });
    context.subscriptions.push(generateCommandDisposable, acceptCommandDisposable);
    this._disposables.push(generateCommandDisposable, acceptCommandDisposable);

    const generateTabCommandDisposable = commands.registerCommand(COMMANDS.GENERATE_INLINE_COPMLETION_TAB, () => {
      setIsGeneralTabActive(true);
      generateFunction();
    });
    context.subscriptions.push(generateTabCommandDisposable, acceptCommandDisposable);
    this._disposables.push(generateTabCommandDisposable, acceptCommandDisposable);
  }

  deactivate(): void {
    this._disposables.forEach((disposable) => {
      disposable.dispose();
    });
    this._disposables = [];
  }
}

export const inlineCompletion = new InlineCompletion();
