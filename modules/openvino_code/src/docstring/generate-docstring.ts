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

    // fixme: docstring parts doesn't includes decorators
    const docstringParts = parse(document, position.line);
    const indentation = getDocstringIndentation(
      document,
      position.line,
      getDefaultIndentation(this.editor.options.insertSpaces as boolean, this.editor.options.tabSize as number)
    );

    return this._insertGenerationPlaceholder(indentation)
      .then(() => this._insertDocstring(docstringParts, indentation, position))
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

  private _insertDocstring(docstringParts: DocstringParts, indentation: string, position: vs.Position) {
    const removeGenerationPlaceholder = () =>
      this.editor.edit((builder) => {
        builder.delete(this.editor.document.lineAt(position).range);
      });

    return backendService
      .generateSummarization({
        inputs: docstringParts.code.join(' '),
        parameters: {
          temperature: extensionState.config.temperature,
          top_k: extensionState.config.topK,
          top_p: extensionState.config.topP,
          min_new_tokens: extensionState.config.minNewTokens,
          max_new_tokens: extensionState.config.maxNewTokens,
        },
      })
      .then((response) => {
        docstringParts.summary = response?.generated_text || '';
        const docstringSnippet = this.generateDocstringSnippet(docstringParts, indentation);

        return removeGenerationPlaceholder().then(() =>
          this.editor.insertSnippet(docstringSnippet, position.with(position.line, 0))
        );
      });
  }

  private generateDocstringSnippet(docstringParts: DocstringParts, indentation: string): vs.SnippetString {
    const config = extensionState.config;

    const docstringFactory = new DocstringFactory(
      getTemplate(extensionState.config.docstringFormat),
      config.quoteStyle,
      true,
      false,
      false,
      true
    );

    const docstring = docstringFactory.generateDocstring(docstringParts, indentation);

    return new vs.SnippetString(docstring);
  }
}
