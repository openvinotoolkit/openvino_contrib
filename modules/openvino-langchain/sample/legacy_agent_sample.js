import { basename } from 'node:path';
import { ChatOpenVINO } from 'openvino-langchain';
import { createToolCallingAgent, AgentExecutor } from 'langchain/agents';
import { ChatPromptTemplate } from "@langchain/core/prompts";
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
});

const tools = [getWeatherTool];
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant"],
  ["placeholder", "{chat_history}"],
  ["human", "{input}"],
  ["placeholder", "{agent_scratchpad}"],
]);
const agent = await createToolCallingAgent({ llm, tools, prompt });
const agentExecutor = new AgentExecutor({
  agent,
  tools,
});
const result = await agentExecutor.invoke({ input: "whats the weather in sf?" });

console.log(result);
