import { test, expect } from '@jest/globals';
import { OpenVINOEmbeddings } from '../embeddings.js';

const modelPath = process.env.EMBEDDING_MODEL_PATH;
if (modelPath === undefined)
  throw new Error('EMBEDDING_MODEL_PATH is not defined');

test('Test embedQuery', async () => {
  const embeddings = new OpenVINOEmbeddings({
    modelPath,
  });
  const res = await embeddings.embedQuery('Hello world');
  expect(typeof res[0]).toBe('number');
});

test('Test embedDocuments', async () => {
  const embeddings = new OpenVINOEmbeddings({
    modelPath,
  });
  const res = await embeddings.embedDocuments([
    'Hello world',
    'Bye bye',
    'we need',
    'at least',
    'six documents',
    'to test pagination',
  ]);
  expect(res).toHaveLength(6);
  res.forEach((r) => {
    expect(typeof r[0]).toBe('number');
  });
});
