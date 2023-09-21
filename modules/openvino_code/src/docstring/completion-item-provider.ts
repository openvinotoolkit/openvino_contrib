import { CompletionItemProvider, CompletionItem, TextDocument, Position, CompletionItemKind, Range } from 'vscode';
import { validDocstringPrefix, docstringIsClosed } from './parse';
import { extensionState } from '../state';
import { COMMANDS } from '../constants';

export const completionItemProvider: CompletionItemProvider = {
  provideCompletionItems: (document: TextDocument, position: Position) => {
    const { isSummarizationSupported } = extensionState.state.features;
    if (isSummarizationSupported && validEnterActivation(document, position)) {
      return [new AutoDocstringCompletionItem(document, position)];
    }
    return;
  },
};

/**
 * Checks that the preceding characters of the position is a valid docstring prefix
 * and that the prefix is not part of an already closed docstring
 */
function validEnterActivation(document: TextDocument, position: Position): boolean {
  const docString = document.getText();
  const quoteStyle = getQuoteStyle();
  return (
    validDocstringPrefix(docString, position.line, position.character, quoteStyle) &&
    !docstringIsClosed(docString, position.line, position.character, quoteStyle)
  );
}

/**
 * Completion item to trigger generate docstring command on docstring prefix
 */
class AutoDocstringCompletionItem extends CompletionItem {
  constructor(_: TextDocument, position: Position) {
    super('Generate Docstring', CompletionItemKind.Snippet);
    this.insertText = '';
    this.filterText = getQuoteStyle();
    this.sortText = '\0';

    this.range = new Range(new Position(position.line, 0), position);

    this.command = {
      command: COMMANDS.GENERATE_DOC_STRING,
      title: 'Generate Docstring',
    };
  }
}

function getQuoteStyle(): string {
  return extensionState.config.quoteStyle || '"""';
}
