import { basename } from 'node:path';
import * as https from 'https';
import { HttpsProxyAgent } from 'https-proxy-agent';
import { ChatOpenVINO } from 'openvino-langchain';
import { tool } from '@langchain/core/tools';

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

const query = 'What is the weather in London?';

const llmWithTools = llm.bindTools([get_weather_tool]);
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
