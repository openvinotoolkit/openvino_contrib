import { join } from 'node:path';
import { addon as ov } from 'openvino-node';
// CVS-146344
type CompiledModel = {
  createInferRequest: () => InferRequest;
  output: (index: number) => any;
  outputs: any[];
  inputs: any[];
};
type InferRequest = {
  inferAsync: (inputs: any) => Promise<{ [outputName: string]: Tensor }>;
};
type Tensor = {
  getShape: () => number[];
  data: SupportedTypedArray;
};

type SupportedTypedArray =
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

// Could not find a declaration file for module.
// It will be fixed in the next release.
// @ts-ignore
import { path as tokenizerExtensionPath } from 'openvino-tokenizers-node';
import { Embeddings, EmbeddingsParams } from '@langchain/core/embeddings';

export interface OvEmbeddingsParams extends EmbeddingsParams {
  modelPath: string;
  device?: string;
}

export class OpenVINOEmbeddings extends Embeddings {

  private tokenizerModelCompiled: Promise<CompiledModel>;

  private modelCompiled: Promise<CompiledModel>;

  constructor(fields: OvEmbeddingsParams) {
    super(fields);
    const modelPath = join(fields.modelPath, 'openvino_model.xml');
    const tokenizerPath = join(fields.modelPath, 'openvino_tokenizer.xml');
    const core = new ov.Core();
    core.addExtension(tokenizerExtensionPath);
    const device = fields?.device || 'CPU';
    this.tokenizerModelCompiled = core.readModel(tokenizerPath)
      .then(result => core.compileModel(result, device))
      .catch((err) => {
        console.error('Failed to compile tokenizer model:', err);
        throw err;
      });
    this.modelCompiled = core.compileModel(modelPath, device)
      .catch((err) => {
        console.error('Failed to compile main model:', err);
        throw err;
      });
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    const tokensArray = [];

    for (const text of texts) {
      const encodings = await this.ovCall(text);

      tokensArray.push(encodings);
    }

    const embeddings = [];

    for (const tokens of tokensArray) {
      const embedArray = [];

      for (const token of tokens) {
        embedArray.push(+token);
      }

      embeddings.push(embedArray);
    }

    return embeddings;
  }

  async embedQuery(text: string): Promise<number[]> {
    const tokens = [];
    const encodings = await this.ovCall(text);

    for (const token of encodings) {
      tokens.push(+token);
    }

    return tokens;
  }

  async ovCall(text: string) {
    const tokenizerModelCompiled = await this.tokenizerModelCompiled;
    const irTokenizer = tokenizerModelCompiled.createInferRequest();

    const inputTensor = new ov.Tensor([text]);
    const tokenizedInput = await irTokenizer.inferAsync([inputTensor]);

    const modelCompiled = await this.modelCompiled;

    const tokenizerOutputs = tokenizerModelCompiled.outputs
      .map(o => o.getAnyName())
      .sort();
    const embeddingInputs = modelCompiled.inputs
      .map(i => i.getAnyName())
      .sort();
    if (!tokenizerOutputs.every(
      (value, index) => value === embeddingInputs[index],
    )) {
      throw Error('Embedding model is incorrect. Tokenizer outputs should be the same as embedding model inputs.');
    }
    const ir = modelCompiled.createInferRequest();
    const embeddings = await ir.inferAsync(tokenizedInput);

    const outputName = modelCompiled.output(0);

    return embeddings[outputName].data;
  }
}
