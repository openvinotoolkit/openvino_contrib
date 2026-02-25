import {
  BaseChatModelParams,
  BindToolsInput,
  ToolChoice,
} from '@langchain/core/language_models/chat_models';
import {
  AIMessage,
  BaseMessage,
  InvalidToolCall,
  SystemMessage,
} from '@langchain/core/messages';
import {
  GenerationConfig,
} from 'openvino-genai-node';
import { ToolCall } from '@langchain/core/messages/tool';
import { convertToOpenAITool } from '@langchain/core/utils/function_calling';

export interface ChatOpenVINOInput extends BaseChatModelParams {
  generationConfig?: GenerationConfig,
  modelPath: string,
  device?: string,
}

export function getValidTools(
  tools: BindToolsInput[] | undefined,
  tool_choice: ToolChoice,
): BindToolsInput[] {
  if (!tools || tool_choice === 'none') {
    return [];
  }
  let usedTools = tools;
  if (tool_choice !== 'auto' && tool_choice !== 'any') {
    if (typeof tool_choice === 'object') {
      tool_choice = tool_choice.name ?? tool_choice.function.name;
    }
    usedTools = usedTools.filter((tool) => {
      if ('name' in tool) {
        return tool.name === tool_choice;
      }
      if ('function' in tool) {
        return tool.function.name === tool_choice;
      }

      return false;
    });
  }

  return usedTools;
}

export interface IToolParser {
  buildPrompt(
    pipeline: any,
    messageHistory: BaseMessage[],
    tools: BindToolsInput[],
    tool_choice?: ToolChoice,
  ): string;
  parseLLMResponse(text: string): AIMessage;
}

export class BaseToolParser implements IToolParser {
  constructor() {}

  protected convertMessageRole(role: string): string {
    switch (role) {
    case 'system':
      return 'system';
    case 'user':
    case 'human':
      return 'user';
    case 'assistant':
    case 'ai':
      return 'assistant';
    case 'tool':
      return 'tool';
    default:
      return 'user';
    }
  }

  buildPrompt(
    pipeline: any,
    messageHistory: BaseMessage[],
    tools: BindToolsInput[],
    tools_choice?: ToolChoice,
  ): string {
    return pipeline.getTokenizer().applyChatTemplate(
      [
        ...this.toolConvert(tools, tools_choice),
        ...messageHistory.map(m => this.messageConvert(m)),
      ].map(m => ({
        role: this.convertMessageRole(m.getType()),
        content: m.content,
      })),
      true,
      '');
  }

  protected messageConvert(message: BaseMessage): BaseMessage {
    return message;
  }

  protected toolConvert(
    tools: BindToolsInput[] | undefined,
    tools_choice?: ToolChoice,
  ): SystemMessage[] {
    if (!tools || tools_choice === 'none') {
      return [];
    }
    let usedTools = getValidTools(tools, tools_choice ?? 'auto');
    const toolMessages: SystemMessage[] = [new SystemMessage(`
# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
${usedTools
    .map((tool) => JSON.stringify(convertToOpenAITool(tool)))
    .join('\n')}
</tools>
For each function call, return a json object with function name and ` +
`arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
`)];

    if (tools_choice === 'any') {
      toolMessages.push(
        new SystemMessage('You must use at least one of the tools.'),
      );
    }

    return toolMessages;
  }

  parseLLMResponse(result: string, max_tool_calls?: number): AIMessage {
    let text = result;
    let toolInfo = '';

    const toolCalls: ToolCall[] = [];
    const invalidToolCalls: InvalidToolCall[] = [];
    let i = text.indexOf('<tool_call>');
    while (i !== -1) {
      if (max_tool_calls && toolCalls.length >= max_tool_calls) {
        break;
      }
      const j = text.indexOf('</tool_call>', i);
      if (0 <= i && i < j) {
        // If the text has `<tool_call>` and `</tool_call>`,
        try {
          toolInfo = text.slice(i + '<tool_call>'.length, j).trim();
          const toolCall = JSON.parse(toolInfo);
          toolCalls.push({
            name: toolCall.name,
            args: toolCall.arguments,
            type: 'tool_call',
          });
        } catch(error: any) {
          // Ignore errors and continue
          invalidToolCalls.push({
            args: toolInfo,
            error: `Failed to parse tool call: ${error.message}`,
            type: 'invalid_tool_call',
          });
        } finally {
          text = (text.slice(0, i) +
            text.slice(j + '</tool_call>'.length)).trim();
          i = text.indexOf('<tool_call>', i);
        }
      } else {
        break;
      }
    }

    return new AIMessage({
      content: text,
      tool_calls: toolCalls,
      invalid_tool_calls: invalidToolCalls,
    });
  }
}
