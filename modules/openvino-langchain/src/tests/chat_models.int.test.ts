import { test, expect, beforeAll } from '@jest/globals';
import { ChatOpenVINO } from '../chat_models.js';
import { BaseMessage, HumanMessage } from '@langchain/core/messages';
import { GenerationConfig } from 'openvino-genai-node';
import { describe } from 'node:test';
import { z } from 'zod';

const modelPath = process.env.MODEL_PATH;
if (modelPath === undefined) throw new Error('MODEL_PATH is not defined');

describe('Test ChatOpenVINO', () => {
  let modelWithBaseConfig: ChatOpenVINO;

  const baseConfig: GenerationConfig = {
    'max_new_tokens': 20,
  };

  beforeAll(() => {
    modelWithBaseConfig = new ChatOpenVINO({
      modelPath,
      generationConfig: baseConfig,
    });
  });

  /**
    * invoke() tests
    */
  {
    test('invoke()', async () => {
      const res = await modelWithBaseConfig.invoke('what is 1 + 1?');
      expect(res).toBeTruthy();
      expect(res instanceof BaseMessage);
      expect(res.content).toEqual(res.text);
    });

    test('invoke() with stop', async () => {
      const res = await modelWithBaseConfig.invoke([
        ['human', 'what is 1 + 1?'],
      ], {
        stop: ['2', 'two'],
      });
      expect(res).toBeTruthy();
      expect(res.text.endsWith('2') || res.text.endsWith('two')).toBe(true);

    });

    test('invoke() with handleLLMNewToken callback', async () => {
      const tokens: string[] = [];
      const res = await modelWithBaseConfig.invoke(
        [new HumanMessage('what is 1 + 1?')],
        {
          callbacks: [
            {
              handleLLMNewToken(token) {
                tokens.push(token);
              },
            },
          ],
        },
      );
      expect(tokens.length).toBeGreaterThanOrEqual(1);
      expect(tokens.join('')).toBe(res.text);
    });

    test('invoke() is interrupted by the abort signal', async () => {
      await expect(() => modelWithBaseConfig.invoke(
        'what is 1 + 1?',
        {
          signal: AbortSignal.timeout(1),
        },
      ),
      ).rejects.toThrow();
    });

    test('invoke() is interrupted by timeout', async () => {
      await expect(() => modelWithBaseConfig.invoke(
        'what is 1 + 1?',
        {
          timeout: 1,
        },
      ),
      ).rejects.toThrow();
    });

  }

  /**
    * stream() tests
    */
  {
    test('stream()', async () => {
      let res = '';
      const streamer = await modelWithBaseConfig.stream('what is 1 + 1?');
      for await (const streamItem of streamer) {
        res += streamItem.text;
        expect(streamItem instanceof BaseMessage);
        expect(streamItem.content).toEqual(streamItem.text);
      }
      expect(res).toBeTruthy();
    });

    test.skip('stream() with stop', async () => {
      let chunks: string[] = [];
      const streamer = await modelWithBaseConfig.stream(
        'what is 1 + 1?',
        {
          stop: ['2', 'two'],
        },
      );
      for await (const streamItem of streamer) {
        chunks.push(streamItem.text);
      }
      expect(chunks.length).toBeGreaterThan(0);
      expect(chunks[chunks.length - 1].includes('2')
        || chunks[chunks.length - 1].includes('two')).toBe(true);
    });

    test('stream() with handleLLMNewToken callback', async () => {
      const tokens: string[] = [];
      let responseContent = '';
      const streamer = await modelWithBaseConfig.stream(
        [new HumanMessage('what is 1 + 1?')],
        {
          callbacks: [
            {
              handleLLMNewToken(token) {
                tokens.push(token);
              },
            },
          ],
        },
      );
      for await (const streamItem of streamer) {
        responseContent += streamItem.text;
      }
      expect(tokens.length).toBeGreaterThanOrEqual(1);
      expect(tokens.join('')).toBe(responseContent);
    });

    // stream doesn't support a signal yet
    test.skip('stream() is interrupted by the abort signal', async () => {
      const streamer = await modelWithBaseConfig.stream(
        [new HumanMessage('what is 1 + 1?')],
        {
          signal: AbortSignal.timeout(1),
        },
      );
      expect(async () => {
        const tokens: string[] = [];
        for await (const streamItem of streamer) {
          tokens.push(streamItem.text);
        }
      }).rejects.toThrow();
    });

    // stream doesn't support a timeout yet
    test.skip('stream() is interrupted by timeout', async () => {
      const streamer = await modelWithBaseConfig.stream(
        [new HumanMessage('what is 1 + 1?')],
        {
          timeout: 1,
        },
      );
      expect(async () => {
        const tokens: string[] = [];
        for await (const streamItem of streamer) {
          tokens.push(streamItem.text);
        }
      }).rejects.toThrow();
    });

    // It will be fixed in the next release of openvino-genai-node
    test.skip('stream() is interrupted by early break', async () => {
      const tokens: string[] = [];
      const streamer = await modelWithBaseConfig.stream(
        [new HumanMessage('what is 1 + 1?')],
      );
      for await (const streamItem of streamer) {
        tokens.push(streamItem.text);
        if (tokens.length >= 5) {
          break;
        }
      }
      expect(tokens.length).toEqual(5);
    });
  }

  /**
    * other API
    */
  {
    test('Using a system message', async () => {
      const res = await modelWithBaseConfig.invoke([
        ['system', 'You are an amazing translator.'],
        ['human', 'Translate "I love programming" into Korean.'],
      ]);
      expect(res).toBeTruthy();
      expect(res instanceof BaseMessage);
    });

    test('generate()', async () => {
      const res = await modelWithBaseConfig.generate([
        ['human', 'what is 1 + 1?'],
      ]);
      expect(res).toBeTruthy();
    });

    // ChatOpenVINO doesn't support a structure output yet
    test.skip('ChatOpenVINO with withStructuredOutput', async () => {
      const modelWithTools = modelWithBaseConfig.withStructuredOutput(
        z.object({
          zomg: z.string(),
          omg: z.number().optional(),
        }),
      );
      const prompt = new HumanMessage(
        'Search the web and tell me what the weather '
        + 'will be like tonight in new york. use weather.com',
      );
      const res = await modelWithTools.invoke([prompt]);
      expect(typeof res.zomg === 'string').toBe(true);
    });
  }
});
