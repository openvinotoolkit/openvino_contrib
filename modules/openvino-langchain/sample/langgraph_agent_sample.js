import { basename } from 'node:path';
import { ChatOpenVINO } from 'openvino-langchain';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { HumanMessage } from '@langchain/core/messages';
import { getWeatherTool, generateImageTool } from './tools.js';

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

function printAgentMessages(messages) {
  for (const message of messages) {
    printAgentMessage(message);
  }
}

function printAgentMessage(message) {
  if (message.getType() === 'tool') {
    console.log(`Tool ${message.name} response: ${message.text}`);

    return;
  }
  if (message.getType() === 'human') {
    console.log(`Human says: ${message.text}`);

    return;
  }
  if (message.getType() === 'ai') {
    if (message.tool_calls && message.tool_calls.length > 0) {
      console.log(
        `AI ${message.text ? 'thought: "' + message.text + '" and ' : ''
        }decided to call tool "${message.tool_calls[0].name
        }" with args ${JSON.stringify(message.tool_calls[0].args)
        }`);
    } else if (
      message.invalid_tool_calls && message.invalid_tool_calls.length > 0
    ) {
      console.log(
        `AI ${message.text ? 'thought: "' + message.text + '" and ' : ''
        }tried to call tool "${message.invalid_tool_calls[0].name
        }" with args ${JSON.stringify(message.invalid_tool_calls[0].args)
        }, but the call was invalid: ${message.invalid_tool_calls[0].error
        }`);
    } else {
      console.log(`AI response: ${message.text}`);
    }

    return;
  }
  console.log(message.text);
}

const query = 'get the weather in London, and create a picture of Big Ben '
  + 'based on the weather information';
const tools = [getWeatherTool, generateImageTool];

const appWithMessagesModifier = createReactAgent({
  llm,
  tools,
});

const stream = await appWithMessagesModifier.stream({
  messages: [new HumanMessage(query)],
});
for await (const chunk of stream) {
  if ('agent' in chunk) {
    printAgentMessages(chunk.agent.messages);
  } else if ('tools' in chunk) {
    printAgentMessages(chunk.tools.messages);
  } else {
    console.log(chunk);
  }
}

// You can also use `invoke` to get the final result without streaming

// const result = await appWithMessagesModifier.invoke({
//     messages: [new HumanMessage(query)],
// });

// printAgentMessage(result.messages[result.messages.length - 1]);
