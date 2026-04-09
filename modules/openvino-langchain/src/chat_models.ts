import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import {
  BaseLanguageModelInput,
} from '@langchain/core/language_models/base';
import {
  BaseChatModel,
  BaseChatModelCallOptions,
  BaseChatModelParams,
  BindToolsInput,
  ToolChoice,
} from '@langchain/core/language_models/chat_models';
import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  InvalidToolCall,
} from '@langchain/core/messages';
import { ChatGenerationChunk, ChatResult } from '@langchain/core/outputs';
import {
  GenerationConfig,
  LLMPipeline,
  StreamingStatus,
} from 'openvino-genai-node';
import { Runnable } from '@langchain/core/runnables';
import { convertToOpenAITool } from '@langchain/core/utils/function_calling';
import {
  BaseToolParser,
  getValidTools,
  IToolParser,
} from './prompt_parser.js';
import { ToolCall } from '@langchain/core/messages/tool';

export interface ChatOpenVINOInput extends BaseChatModelParams {
  generationConfig?: GenerationConfig,
  modelPath: string,
  device?: string,
  toolParser?: IToolParser,
}

export interface ChatOpenVINOCallOptions
  extends BaseChatModelCallOptions {
  tools?: BindToolsInput[];
  tool_choice?: ToolChoice;
}

export class ChatOpenVINO
  extends BaseChatModel<ChatOpenVINOCallOptions>
  implements ChatOpenVINOInput
{
  generateOptions: GenerationConfig;

  modelPath: string;

  device: string;

  pipeline: Promise<any>;

  toolParser: IToolParser;

  constructor(params: ChatOpenVINOInput) {
    super(params);
    this.modelPath = params.modelPath;
    this.device = params.device || 'CPU';
    this.pipeline = LLMPipeline(this.modelPath, this.device);
    this.generateOptions = params.generationConfig || {};
    this.toolParser = new BaseToolParser();
  }

  static lc_name() {
    return 'ChatOpenVINO';
  }

  _llmType() {
    return 'OpenVINO';
  }

  private updateToolCalls(
    message: AIMessage,
    tools: BindToolsInput[],
    tool_choice?: ToolChoice,
  ): AIMessage {
    const validToolNames = getValidTools(tools, tool_choice ?? 'auto')
      .map(tool => 'name' in tool ? tool.name : tool.function.name);
    const toolCalls: ToolCall[] = [];
    const invalidToolCalls: InvalidToolCall[] = [];
    for (const toolCall of message.tool_calls || []) {
      if (validToolNames.includes(toolCall.name)) {
        toolCalls.push(toolCall);
      } else {
        invalidToolCalls.push({
          name: toolCall.name,
          args: JSON.stringify(toolCall.args),
          id: toolCall.id,
          error: `Tool "${toolCall.name}" is not allowed. Valid tools are: `
            + validToolNames.join(', '),
        });
      }
    }

    return new AIMessage({
      ...message,
      tool_calls: toolCalls,
      invalid_tool_calls: invalidToolCalls,
    });
  }

  override bindTools(
    tools: BindToolsInput[],
    kwargs?: Partial<this['ParsedCallOptions']>,
  ): Runnable<BaseLanguageModelInput, AIMessageChunk, ChatOpenVINOCallOptions> {
    return this.withConfig({
      tools: tools.map((tool) => convertToOpenAITool(tool)),
      ...kwargs,
    });
  }

  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    if (!messages.length) {
      throw new Error('No messages provided.');
    }
    if (typeof messages[0].content !== 'string') {
      throw new Error('Multimodal messages are not supported.');
    }
    const pipeline = await this.pipeline;

    // Signal setup
    const signals: AbortSignal[] = [];
    if (options.signal) {
      signals.push(options.signal);
    }
    if (options.timeout) {
      signals.push(AbortSignal.timeout(options.timeout));
    }
    const signal = AbortSignal.any(signals);

    // generation option setup
    const generateOptions: GenerationConfig = { ...this.generateOptions };
    // to avoid a warning about result type
    generateOptions['return_decoded_results'] = true;
    if (options.stop) {
      const set = new Set(options.stop);
      generateOptions['stop_strings'] = set;
      generateOptions['include_stop_str_in_output'] = true;
    }

    // callback setup
    const callback = (chunk: string) => {
      runManager?.handleLLMNewToken(chunk).catch(console.error);

      return signal.aborted ? StreamingStatus.CANCEL : StreamingStatus.RUNNING;
    };

    const prompt = this.toolParser.buildPrompt(
      pipeline,
      messages,
      options.tools ?? [],
      options.tool_choice,
    );

    const result = await pipeline.generate(
      prompt,
      generateOptions,
      callback,
    );
    // We need to throw an exception if the generation was canceled by a signal
    signal.throwIfAborted();

    return {
      generations: result.texts.map((r: string) => {
        const message = this.toolParser.parseLLMResponse(r);

        return {
          text: message.content,
          message: this.updateToolCalls(
            message,
            options.tools ?? [],
            options.tool_choice,
          ),
        };
      }),
    };
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    if (!messages.length) {
      throw new Error('No messages provided.');
    }
    const pipeline = await this.pipeline;
    const prompt = this.toolParser.buildPrompt(
      pipeline,
      messages,
      options.tools ?? [],
      options.tool_choice,
    );
    const generator = pipeline.stream(
      prompt,
      this.generateOptions,
    );

    let chunk = await generator.next();
    while (!chunk.done) {
      yield new ChatGenerationChunk({
        message: new AIMessageChunk({
          content: chunk.value,
        }),
        text: chunk.value,
      });
      await runManager?.handleLLMNewToken(chunk.value);
      chunk = await generator.next();
    }
    // pipeline.stream returns a full result at the end
    // so we are able to parse tool calls from it
    let aiMessage = this.toolParser.parseLLMResponse(
      chunk.value.subword,
    );
    aiMessage = this.updateToolCalls(
      aiMessage,
      options.tools ?? [],
      options.tool_choice,
    );
    if (aiMessage.tool_calls?.length) {
      yield new ChatGenerationChunk({
        message: new AIMessageChunk({
          content: '',
          tool_calls: aiMessage.tool_calls,
          tool_call_chunks: aiMessage.tool_calls.map((tc, index) => ({
            name: tc.name,
            args: JSON.stringify(tc.args),
            index,
            id: tc.id,
          })),
        }),
        text: '',
      });
    }
  }
}
