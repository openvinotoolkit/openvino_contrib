import * as vs from 'vscode';
import { DocstringFactory } from './docstring-template/docstring-factory';
import { getTemplate } from './docstring-template/get-template';
import { getDocstringIndentation, getDefaultIndentation, parse } from './parse';
import { DocstringParts } from './docstring-template/docstring-parts';
import { backendService } from '../services/backend.service';
import { extensionState } from '../state';

export class AutoDocstring {
  private editor: vs.TextEditor;

  constructor(editor: vs.TextEditor) {
    this.editor = editor;
  }

  public generate() {
    extensionState.set('isLoading', true);

    const position = this.editor.selection.active;
    const document = this.editor.document.getText();

    const defaultIndentation = getDefaultIndentation(
      this.editor.options.insertSpaces as boolean,
      this.editor.options.tabSize as number
    );
    const { docstringParts, definition } = parse(document, position.line, defaultIndentation.length);
    const indentation = getDocstringIndentation(document, position.line, defaultIndentation);

    return this._insertGenerationPlaceholder(indentation)
      .then(() => this._generateDocstring(docstringParts, definition, indentation, position))
      .then(
        () => extensionState.set('isLoading', false),
        () => extensionState.set('isLoading', false)
      );
  }

  private _insertGenerationPlaceholder(indentation: string) {
    const position = this.editor.selection.active;
    const insertPosition = position.with(position.line, 0);

    const quoteStyle = extensionState.config.quoteStyle || '"""';
    const generationPlaceholderSnippet = new vs.SnippetString(
      `${indentation}${quoteStyle} Generating summarization ${quoteStyle}`
    );
    return this.editor.insertSnippet(generationPlaceholderSnippet, insertPosition);
  }

  private _removeGenerationPlaceholder(position: vs.Position) {
    return this.editor.edit((builder) => {
      builder.delete(this.editor.document.lineAt(position).range);
    });
  }

  private _generateDocstring(
    docstringParts: DocstringParts,
    definition: string,
    indentation: string,
    position: vs.Position
  ) {
    const template = this.generateTemplate(docstringParts, indentation);

    return backendService
      .generateSummarization({
        inputs: docstringParts.code.join('\n'),
        template: template,
        definition: definition,
        format: extensionState.config.docstringFormat,
        parameters: {
          temperature: extensionState.config.temperature,
          top_k: extensionState.config.topK,
          top_p: extensionState.config.topP,
          min_new_tokens: extensionState.config.minNewTokens,
          max_new_tokens: extensionState.config.maxNewTokens,
          timeout: extensionState.config.serverRequestTimeout,
          repetition_penalty: extensionState.config.repetitionPenalty,
        },
      })
      .then((response) => {
        const docstringSnippet = new vs.SnippetString(response?.generated_text);

        return this._removeGenerationPlaceholder(position).then(() =>
          this.editor.insertSnippet(docstringSnippet, position.with(position.line, 0))
        );
      });
  }

  private generateTemplate(docstringParts: DocstringParts, indentation: string): string {
    const { quoteStyle, docstringFormat } = extensionState.config;

    const docstringFactory = new DocstringFactory(getTemplate(docstringFormat), quoteStyle, true, false, false, true);

    return docstringFactory.generateDocstring(docstringParts, indentation);
  }
}
