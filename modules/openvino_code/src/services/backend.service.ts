import fetch, { AbortError, FetchError, RequestInit } from 'node-fetch';
import { extensionState } from '../state';
import { notificationService } from './notification.service';
import { lruCache } from '../lru-cache.decorator';
import { ConnectionStatus } from '@shared/extension-state';
import { streamingRequest } from './request';

export interface IGenerateRequest {
  inputs: string;
  parameters: {
    temperature: number;
    top_k: number;
    top_p: number;
    min_new_tokens: number;
    max_new_tokens: number;
    timeout: number;
    repetition_penalty: number;
  };
}

interface IGenerateDocStringRequest {
  inputs: string;
  template: string;
  definition: string;
  format?: string;
  parameters: {
    temperature: number;
    top_k: number;
    top_p: number;
    min_new_tokens: number;
    max_new_tokens: number;
    timeout: number;
    repetition_penalty: number;
  };
}

interface IGenerateResponse {
  generated_text: string;
}

type RequestMethodType = 'GET' | 'POST';

interface RequestOptions {
  timeout: number;
}

class ServerError extends Error {}

const skipEmptyGeneratedText = (response: IGenerateResponse | null) => !response?.generated_text.trim();

class BackendService {
  private readonly _apiSlug = 'api';

  private get _apiUrl(): string {
    return `${extensionState.config.serverUrl}/${this._apiSlug}`;
  }

  private get _endpoints() {
    return {
      health: `${this._apiUrl}/health`,
      generate: `${this._apiUrl}/generate`,
      summarize: `${this._apiUrl}/summarize`,
    };
  }

  private readonly _headers = {
    'Content-Type': 'application/json',
    Authorization: '',
  };

  private get _requestTimeoutMs(): number {
    return extensionState.config.serverRequestTimeout * 1000;
  }

  async healthCheck(): Promise<object | null> {
    return this._sendRequest(this._endpoints.health, 'GET', null, { timeout: this._requestTimeoutMs * 2 });
  }

  @lruCache<IGenerateResponse>({ skipAddToCache: skipEmptyGeneratedText })
  async generateCompletion(data: IGenerateRequest): Promise<IGenerateResponse | null> {
    return this._sendRequest<IGenerateRequest, IGenerateResponse>(this._endpoints.generate, 'POST', data);
  }

  async generateCompletionStream(
    data: IGenerateRequest,
    onDataChunk: (chunk: string) => void,
    signal?: AbortSignal
  ): Promise<void> {
    return streamingRequest(`${this._apiUrl}/generate_stream`, onDataChunk, {
      method: 'POST',
      timeout: this._requestTimeoutMs,
      headers: this._headers,
      body: data,
      signal: signal,
    });
  }

  async generateSummarization(data: IGenerateDocStringRequest): Promise<IGenerateResponse | null> {
    return this._sendRequest<IGenerateRequest, IGenerateResponse>(this._endpoints.summarize, 'POST', data);
  }

  private async _sendRequest<T, R>(
    url: string,
    method: RequestMethodType,
    data?: T,
    options: RequestOptions = { timeout: this._requestTimeoutMs }
  ): Promise<R | null> {
    const controller = new AbortController();

    const abortTimeout = setTimeout(() => {
      controller.abort();
    }, options.timeout);

    return fetch(url, {
      method,
      headers: this._headers,
      body: data ? JSON.stringify(data) : null,
      signal: controller.signal as unknown as RequestInit['signal'],
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new ServerError(`Error during sending request to ${url} (${response.status} ${response.statusText})`);
        }
        return (await response.json()) as R;
      })
      .catch((error) => {
        if (error instanceof AbortError) {
          console.error('Request was timed out.');
          notificationService.showRequestTimeoutMessage();
        }
        if (error instanceof FetchError && url !== this._endpoints.health) {
          extensionState.set('connectionStatus', ConnectionStatus.NOT_AVAILABLE);
          notificationService.showServerNotAvailableMessage(extensionState.state);
        }
        if (error instanceof ServerError) {
          notificationService.showWarningMessage(error.message);
        }
        console.error(error);
        return null;
      })
      .finally(() => {
        clearTimeout(abortTimeout);
      });
  }
}

export const backendService = new BackendService();
