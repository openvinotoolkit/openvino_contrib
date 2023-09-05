import { getBody, getDefinition, getFunctionName, parseParameters, tokenizeDefinition } from ".";
import { DocstringParts } from "../docstring-template/docstring-parts";

export function parse(document: string, positionLine: number): DocstringParts {
  const definition = getDefinition(document, positionLine);
  const body = getBody(document, positionLine);

  const parameterTokens = tokenizeDefinition(definition);
  const functionName = getFunctionName(definition);
  const code = [definition].concat(body);

  return parseParameters(parameterTokens, body, functionName, code);
}
