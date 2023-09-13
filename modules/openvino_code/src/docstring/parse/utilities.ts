export function getIndentation(line: string): string {
  const whiteSpaceMatches = line.match(/^[^\S\r]+/);

  // fixme: this code relies on implicit type conversion for non strict comparison
  if (whiteSpaceMatches == undefined) {
    return "";
  }

  return whiteSpaceMatches[0];
}

/**
 * Preprocess an array of lines.
 * For example trim spaces and discard comments
 * @param lines The lines to preprocess.
 */
export function preprocessLines(lines: string[]): string[] {
  return lines.map((line) => line.trim()).filter((line) => !line.startsWith("#"));
}

export function filterComments(lines: string[]): string[] {
  return lines.filter((line) => !line.startsWith("#"));
}

export function indentationOf(line: string): number {
  return getIndentation(line).length;
}

export function blankLine(line: string): boolean {
  // fixme: this code relies on implicit type conversion for non strict comparison
  return line.match(/[^\s]/) == undefined;
}

export function getDefaultIndentation(useSpaces: boolean, tabSize: number): string {
  if (!useSpaces) {
    return "\t";
  }

  return " ".repeat(tabSize);
}
