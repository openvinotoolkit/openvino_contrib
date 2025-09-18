import { basename } from 'node:path';
import { ChatOpenVINO } from 'openvino-langchain';
import { getWeatherTool } from './tools.js';

const INSTRUCT_MODEL_PATH = process.argv[2];
const device = 'CPU';

if (!INSTRUCT_MODEL_PATH) {
  console.error('Please specify path to models directories\n'
    + 'Run command must be:\n'
    + `'node ${basename(process.argv[1])} *path_to_llm_model_dir*'`);
  process.exit(1);
}
if (process.argv.length > 3) {
  console.error(
    `Run command must be:
    'node ${basename(process.argv[1])} *path_to_llm_model_dir*'`,
  );
  process.exit(1);
}

const llm = new ChatOpenVINO({
  modelPath: INSTRUCT_MODEL_PATH,
  device,
  generationConfig: {
  },
});

const query = 'What is the weather in London?';

const llmWithTools = llm.bindTools([getWeatherTool]);
const result = await llmWithTools.invoke(query);
console.log(result); // res will contain the tool_calls response

// You also can use streaming
// const stream = await llmWithTools.stream(query);
// let finalChunk;
// for await (const chunk of stream) {
//     if (!finalChunk) {
//         finalChunk = chunk;
//     } else {
//         finalChunk = finalChunk.concat(chunk);
//     }
//     process.stdout.write(chunk.text);
// }
// console.log("\nTool calls:", finalChunk.tool_calls);
// console.log("\nInvalid tool calls:", finalChunk.invalid_tool_calls);
