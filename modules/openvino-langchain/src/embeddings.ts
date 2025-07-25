import {
  EmbeddingResult,
  EmbeddingResults,
  TextEmbeddingPipeline,
} from 'openvino-genai-node';
import { Embeddings, EmbeddingsParams } from '@langchain/core/embeddings';

export interface OvEmbeddingsParams extends EmbeddingsParams {
  modelPath: string;
  device?: string;
}

export class OpenVINOEmbeddings extends Embeddings {
  private pipeline: Promise<any>;

  constructor(fields: OvEmbeddingsParams) {
    super(fields);
    this.pipeline = TextEmbeddingPipeline(
      fields.modelPath,
      fields?.device || 'CPU',
    );
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    const pipeline = await this.pipeline;
    const result: EmbeddingResults = await pipeline.embedDocuments(texts);

    return Array.from(result.map(x => Array.from(x)));
  }

  async embedQuery(text: string): Promise<number[]> {
    const pipeline = await this.pipeline;
    const result: EmbeddingResult = await pipeline.embedQuery(text);

    return Array.from(result);
  }
}
