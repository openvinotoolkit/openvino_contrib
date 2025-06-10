import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { TextLoader } from 'langchain/document_loaders/fs/text';

import { OpenVINO, OpenVINOEmbeddings } from 'openvino-langchain';

// Paths to document and models
const TEXT_DOCUMENT_PATH = './document_sample.txt';
const LLM_MODEL_PATH = process.argv[2];
const EMBEDDINGS_MODEL_PATH = process.argv[3];

if (!LLM_MODEL_PATH || !EMBEDDINGS_MODEL_PATH) {
  console.error('Please specify path to models directories\n'
    + 'Run command must be:\n'
    + '`node index.js *path_to_llm_model_dir* *path_to_embeddings_model_dir*`');
  process.exit(1);
}

// Prompt related parameters: question and template
// const RETRIVER_QUERY = 'largest hot desert';
// const QUESTION = 'What is largest hot desert?';
const RETRIVER_QUERY = 'unique wildlife';
const QUESTION = 'Where is the place with a unique wildlife?';

const PROMPT_TEMPLATE = 'Answer the question based only'
  + 'on the following context: \n{context}\n\nQuestion: {question}';

// Code below is the same as sample from langchain documentation:
// https://js.langchain.com/v0.1/docs/expression_language/cookbook/retrieval/
const loader = new TextLoader(TEXT_DOCUMENT_PATH);
const docs = await loader.load();

// Create instance of OvEmbeddings (it uses openvino-node package inside)
// by passing EMBEDDINGS_MODEL_PATH from script argument
const embeddings = new OpenVINOEmbeddings({ modelPath: EMBEDDINGS_MODEL_PATH });

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const splits = await textSplitter.splitDocuments(docs);
const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);

const retriever = vectorStore.asRetriever();
const prompt = PromptTemplate.fromTemplate(PROMPT_TEMPLATE);

// Create instance of OpenVINO class (it uses openvino-genai-node package
// inside), that implements LLM class methods from llangchain core
// Pass path to LLM model directory and inference parameters into constructor
const llmModel = new OpenVINO({
  modelPath: LLM_MODEL_PATH,
  temperature: 0,
  callbacks: [
    {
      handleLLMNewToken(token) {
        process.stdout.write(token);
      },
    },
  ],
});

const ragChain = await createStuffDocumentsChain({
  llm: llmModel,
  prompt,
  outputParser: new StringOutputParser(),
});

const retrievedDocs = await retriever.invoke(RETRIVER_QUERY);

console.log(`Question: '${QUESTION}'\n`);

await ragChain.invoke({
  question: QUESTION,
  context: retrievedDocs,
});
