import { getBody, getDefinition, getFunctionName, parseParameters, tokenizeDefinition } from '.';
import { DocstringParts } from '../docstring-template/docstring-parts';

export function parse(
  document: string,
  positionLine: number,
  defaultIndentation: number
): { docstringParts: DocstringParts; definition: string } {
  const definition = getDefinition(document, positionLine);
  const body = getBody(document, positionLine, defaultIndentation);

  const parameterTokens = tokenizeDefinition(definition);
  const functionName = getFunctionName(definition);
  const code = [definition, ...body];

  return { docstringParts: parseParameters(parameterTokens, body, functionName, code), definition };
}
