import { test, expect } from '@jest/globals';
import { PromptTemplate } from '@langchain/core/prompts';
import { OpenVINO } from '../llms.js';

const modelPath = process.env.MODEL_PATH;
if (modelPath === undefined) throw new Error('MODEL_PATH doest not defined');

test('Test llm with incorrect modelPath', async () => {
  const model = new OpenVINO({modelPath: ''});
  await expect(model.invoke('1 + 1 =', {})).rejects.toThrow();
});

test('Test llm with incorrect device', async () => {
  const model = new OpenVINO({modelPath, device: 'PC'});
  await expect(model.invoke('1 + 1 =', {})).rejects.toThrow();
});

test('Test invoke', async () => {
  const model = new OpenVINO({
    modelPath,
    generationConfig: {
      'max_new_tokens': 10,
    },
  });
  const res = await model.invoke('1 + 1 =', {});
  expect(res.length).toBeGreaterThan(0);
}, 5000);

test('test invoke with callback', async () => {
  const model = new OpenVINO({
    modelPath,
    generationConfig: {
      'max_new_tokens': 10,
    },
  });
  const tokens: string[] = [];
  const result = await model.invoke(
    'What is a good name for a company that makes colorful socks?',
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
  expect(tokens.length).toBeGreaterThan(1);
  expect(result).toEqual(tokens.join(''));
}, 5000);

test('test invoke with max tokens', async () => {
  const model = new OpenVINO({
    modelPath,
    generationConfig: {
      'max_new_tokens': 10,
    },
  });
  const tokens: string[] = [];
  const result = await model.invoke(
    'What is a good name for a company that makes colorful socks?',
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
  expect(tokens.length).toBeLessThanOrEqual(100);
  expect(result).toEqual(tokens.join(''));
}, 5000);

test('Test stream', async () => {
  const model = new OpenVINO({
    modelPath,
    generationConfig: {
      'max_new_tokens': 10,
    },
  });
  const stream = await model.stream('What is 2 + 2?');
  const chunks: string[] = [];
  for await (const chunk of stream) {
    chunks.push(chunk);
  }
  expect(chunks.length).toBeGreaterThan(1);
}, 5000);

test('Test invoke with abort signal', async () => {
  const model = new OpenVINO({
    modelPath,
  });
  const controller = new AbortController();
  await expect(() => {
    const ret = model.invoke('Respond with an extremely verbose response', {
      signal: controller.signal,
    });
    controller.abort();

    return ret;
  }).rejects.toThrow('This operation was aborted');
}, 5000);

test('Test invoke with stop', async () => {
  const model = new OpenVINO({
    modelPath,
    generationConfig: {
      'max_new_tokens': 100,
    },
  });
  const res = await model.invoke('Print hello world', { stop: ['world'] });
  expect(res.endsWith('world')).toEqual(true);
}, 5000);

test('Test invoke with timeout', async () => {
  const model = new OpenVINO({
    modelPath,
    generationConfig: {
      'max_new_tokens': 10,
    },
  });
  await expect(() =>
    model.invoke('Respond with an extremely verbose response', {
      timeout: 10,
    }),
  ).rejects.toThrow('The operation was aborted due to timeout');
}, 5000);

// TODO: Implement maxConcurrency option
test.skip('Test parallel invoke with concurrency', async () => {
  const model = new OpenVINO({
    modelPath,
    generationConfig: {
      'max_new_tokens': 10,
    },
    maxConcurrency: 1,
  });
  const res = await Promise.all([
    model.invoke('1 + 1 ='),
    model.invoke('2 + 2 ='),
  ]);
  expect(res.length).toBeGreaterThan(0);
}, 5000);

test('Test using model in chain', async () => {
  const TEMPLATE = `You are a pirate named Patchy.
  All responses must be extremely verbose and in pirate dialect.

  User: {input}
  AI:`;

  const prompt = PromptTemplate.fromTemplate(TEMPLATE);
  const model = new OpenVINO({
    modelPath,
    generationConfig: {
      'max_new_tokens': 10,
    },
  });
  const chain = prompt.pipe(model);
  const res = await chain.invoke({'input': '1 + 1 ='});
  expect(res.length).toBeGreaterThan(0);
}, 5000);
