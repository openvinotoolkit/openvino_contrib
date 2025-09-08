import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import { LLM, BaseLLMParams } from '@langchain/core/language_models/llms';
import { GenerationChunk } from '@langchain/core/outputs';

import {
  LLMPipeline,
  GenerationConfig,
  StreamingStatus,
} from 'openvino-genai-node';

export interface OpenVINOParams extends BaseLLMParams {
  generationConfig?: GenerationConfig,
  modelPath: string,
  device?: string,
}

export class OpenVINO extends LLM {
  static lc_name() {
    return 'GenAI';
  }

  generateOptions: GenerationConfig;

  path: string;

  device: string;

  pipeline: Promise<any>;

  constructor(inputs: OpenVINOParams) {
    super(inputs);
    this.generateOptions = inputs.generationConfig || {};
    this.path = inputs.modelPath;
    this.device = inputs.device || 'CPU';
    this.pipeline = LLMPipeline(this.path, this.device);
  }

  _llmType() {
    return 'openvino_genai';
  }

  async _call(
    prompt: string,
    options: this['ParsedCallOptions'],
    runManager: CallbackManagerForLLMRun,
  ): Promise<string> {
    const pipeline = await this.pipeline;

    // Signal setup
    const signals : AbortSignal[] = [];
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
      generateOptions.stop_strings = set;
      generateOptions.include_stop_str_in_output = true;
    }

    // callback setup
    const callback = (chunk: string) => {
      runManager?.handleLLMNewToken(chunk).catch(console.error);

      return signal.aborted ? StreamingStatus.CANCEL : StreamingStatus.RUNNING;
    };

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
    input: string,
    _options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<GenerationChunk> {
    const pipeline = await this.pipeline;
    const generator = pipeline.stream(input, this.generateOptions);
    for await (const chunk of generator) {
      yield new GenerationChunk({
        text: chunk,
      });
      await runManager?.handleLLMNewToken(chunk);
    }
  }
}
