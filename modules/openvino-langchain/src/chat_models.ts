import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import {
  BaseLanguageModelCallOptions,
} from '@langchain/core/language_models/base';
import {
  SimpleChatModel,
} from '@langchain/core/language_models/chat_models';
import { AIMessageChunk, BaseMessage } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import {
  GenerationConfig,
  LLMPipeline,
  StreamingStatus,
} from 'openvino-genai-node';

export interface ChatOpenVINOParams extends BaseLanguageModelCallOptions {
  generationConfig?: GenerationConfig,
  modelPath: string,
  device?: string,
}

export class ChatOpenVINO extends SimpleChatModel {
  generateOptions: GenerationConfig;

  path: string;

  device: string;

  pipeline: Promise<any>;

  constructor(params: ChatOpenVINOParams) {
    super(params);
    this.path = params.modelPath;
    this.device = params.device || 'CPU';
    this.pipeline = LLMPipeline(this.path, this.device);
    this.generateOptions = params.generationConfig || {};
  }
  _llmType() {
    return 'OpenVINO';
  }
  private async convertMessages(messages: BaseMessage[]): Promise<string> {
    const pipeline = await this.pipeline;
    if (messages.length === 0) {
      throw new Error('No messages provided.');
    }

    return pipeline.getTokenizer().applyChatTemplate(
      messages.map(m => ({
        role: m.getType(),
        content: m.text,
      })),
      false,
      '');
  }
  async _call(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): Promise<string> {
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

    const prompt = await this.convertMessages(messages);

    const result = await pipeline.generate(
      prompt,
      generateOptions,
      callback,
    );
    // We need to throw an exception if the generation was canceled by a signal
    signal.throwIfAborted();

    return result.toString();
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    _options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    const pipeline = await this.pipeline;
    const prompt = await this.convertMessages(messages);
    const generator = pipeline.stream(
      prompt,
      this.generateOptions,
    );
    for await (const chunk of generator) {
      yield new ChatGenerationChunk({
        message: new AIMessageChunk({
          content: chunk,
        }),
        text: chunk,
      });
      await runManager?.handleLLMNewToken(chunk);
    }
  }
}
