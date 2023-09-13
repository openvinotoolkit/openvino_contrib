import { blankLine, filterComments, indentationOf } from './utilities';

export function getBody(document: string, linePosition: number, defaultIndentation: number): string[] {
  const lines = document.split('\n');
  const body = [];

  let currentLineNum = linePosition;
  const bodyBaseIndentation = getBodyBaseIndentation(lines, linePosition);

  while (currentLineNum < lines.length) {
    const line = lines[currentLineNum];

    if (blankLine(line)) {
      currentLineNum++;
      continue;
    }

    if (indentationOf(line) < bodyBaseIndentation) {
      break;
    }

    body.push(line.slice(bodyBaseIndentation - defaultIndentation));
    currentLineNum++;
  }

  return filterComments(body);
}

function getBodyBaseIndentation(lines: string[], linePosition: number): number {
  let currentLineNum = linePosition;
  const functionDefRegex = /\s*def \w+/;

  while (currentLineNum < lines.length) {
    const line = lines[currentLineNum];

    if (blankLine(line)) {
      currentLineNum++;
      continue;
    }

    if (functionDefRegex.test(line)) {
      break;
    }

    return indentationOf(line);
  }

  return 10000;
}
