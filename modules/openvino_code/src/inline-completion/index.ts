import { Disposable, ExtensionContext, commands, languages, window } from 'vscode';
import { IExtensionComponent } from '../extension-component.interface';
import { CommandInlineCompletionItemProvider } from './command-inline-completion-provider';
import { COMMANDS } from '../constants';
import { extensionState } from '../state';
import { notificationService } from '../services/notification.service';

class InlineCompletion implements IExtensionComponent {
  activate(context: ExtensionContext): void {
    // Register Inline Completion triggered by command
    const commandInlineCompletionProvider = new CommandInlineCompletionItemProvider();

    let commandInlineCompletionDisposable: Disposable;

    const commandDisposable = commands.registerCommand(COMMANDS.GENERATE_INLINE_COPMLETION, () => {
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
    });

    context.subscriptions.push(commandDisposable);
  }

  deactivate(): void {}
}

export const inlineCompletion = new InlineCompletion();
