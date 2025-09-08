import { tool } from '@langchain/core/tools';
import { basename } from 'node:path';
import * as https from 'https';
import { HttpsProxyAgent } from 'https-proxy-agent';
import { ChatOpenVINO } from 'openvino-langchain';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { HumanMessage } from '@langchain/core/messages';

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

const get_weather_tool = tool(
  async (args) => {
    const { city_name: cityName } = args;
    const keySelection = {
      'current_condition': [
        'temp_C',
        'FeelsLikeC',
        'humidity',
        'weatherDesc',
        'observation_time',
      ],
    };
    const response = new Promise((resolve, reject) => {

      const proxyAgent = process.env.http_proxy
        ? new HttpsProxyAgent(process.env.http_proxy)
        : undefined;

      https.get(
        `https://wttr.in/${cityName}?format=j1`,
        { agent: proxyAgent },
        (res) => {
          let data = '';

          res.on('data', (chunk) => {
            data += chunk.toString();
          });

          res.on('end', () => {
            resolve(JSON.parse(data));
          });

          res.on('error', (err) => {
            reject(err);
          });
        },
      );
    });
    const data = await response;
    const result = {};
    for (const [key, values] of Object.entries(keySelection)) {
      if (data[key] && Array.isArray(data[key]) && data[key][0]) {
        result[key] = {};
        for (const v of values) {
          result[key][v] = data[key][0][v];
        }
      }
    }

    return JSON.stringify(result);
  },
  {
    name: 'get_weather',
    description: 'Get the current weather in a given city name.',
    schema: {
      type: 'object',
      properties: {
        city_name: {
          type: 'string',
          description: 'City name',
        },
      },
      required: ['city_name'],
    },
  },
);

const generate_image_tool = tool(
  async (toolArgs) => {
    const { prompt } = toolArgs;
    const encodedPrompt = encodeURIComponent(prompt);

    return JSON.stringify({
      'image_url': `https://image.pollinations.ai/prompt/${encodedPrompt}`,
    });
  },
  {
    name: 'generate_image',
    description:
            'AI painting (image generation) service, input text description, '
            + 'and return the image URL drawn based on text information.',
    schema: {
      type: 'object',
      properties: {
        prompt: {
          type: 'string',
          description: 'The prompt to generate an image for',
        },
      },
      required: ['prompt'],
    },
  },
);

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
const tools = [get_weather_tool, generate_image_tool];

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
