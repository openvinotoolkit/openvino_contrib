import { blankLine, preprocessLines } from "./utilities";

export function getDefinition(document: string, linePosition: number): string {
  const precedingLines = getPrecedingLines(document, linePosition);
  const precedingText = precedingLines.join(" ");

  // Don't parse if the preceding line is blank
  const precedingLine = precedingLines[precedingLines.length - 1];

  // fixme: this code relies on implicit type conversion for non strict comparison
  if (precedingLine == undefined || blankLine(precedingLine)) {
    return "";
  }

  const pattern = /\b(((async\s+)?\s*def)|\s*class)\b/g;

  // Get starting index of last def match in the preceding text
  let index: number | null = null;
  while (pattern.test(precedingText)) {
    index = pattern.lastIndex - RegExp.lastMatch.length;
  }

  if (index == null) {
    return "";
  }

  const lastFunctionDef = precedingText.slice(index);
  return lastFunctionDef.trim();
}

function getPrecedingLines(document: string, linePosition: number): string[] {
  const lines = document.split("\n");
  const rawPrecedingLines = lines.slice(0, linePosition);

  const precedingLines = preprocessLines(rawPrecedingLines);

  return precedingLines;
}
