import { z } from 'zod';
import {
  expect,
  describe,
  beforeAll,
  test } from '@jest/globals';
import { tool } from '@langchain/core/tools';
import { ChatOpenVINO } from '../chat_models.js';
import { GenerationConfig } from 'openvino-genai-node';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages';

const modelPath = process.env.INSTRUCT_MODEL_PATH;
if (modelPath === undefined)
  throw new Error('export doest not defined');

describe('ChatOpenVINO tools', () => {
  let modelWithBaseConfig: ChatOpenVINO;

  const baseConfig: GenerationConfig = {
  };

  const weatherTool = tool(
    async ({location, unit}) => {
      if (location.toLowerCase().includes('tokyo')) {
        return JSON.stringify({ location, temperature: '10', unit: 'celsius' });
      } else if (location.toLowerCase().includes('san francisco')) {
        return JSON.stringify({
          location,
          temperature: '72',
          unit: 'fahrenheit',
        });
      } else {
        return JSON.stringify({ location, temperature: '22', unit: 'celsius' });
      }
    },
    {
      name: 'get_current_weather',
      description: 'Get the current weather in a given location',
      schema: z.object({
        location: z
          .string()
          .describe('The location to get the current weather for.'),
        unit: z.enum(['celsius', 'fahrenheit']).optional(),
      }),
    },
  );

  beforeAll(() => {
    modelWithBaseConfig = new ChatOpenVINO({
      modelPath,
      generationConfig: baseConfig,
    });
  });

  test.skip('Test ChatOpenVINO tool calling', async () => {
    const chat = modelWithBaseConfig.bindTools([weatherTool]);
    const res = await chat.invoke([
      ['human', 'What\'s the weather like in San Francisco, Tokyo, and Paris?'],
    ]);
    console.log(res);
    expect(res.tool_calls?.length).toEqual(3);
    expect(res.tool_calls?.[0].name).toEqual('get_current_weather');
    expect(res.tool_calls?.every(tooCall =>
      tooCall.args.location.match(/[San Francisco,Tokyo,Paris]+/)),
    ).toBe(true);
    expect(res.tool_calls?.every(tooCall =>
      tooCall.args.unit.match(/^[celsius,fahrenheit]+$/)),
    ).toBe(true);
  }, 150000);

  test.skip('Test ChatOpenVINO tool calling with ToolMessages', async () => {
    const chat = modelWithBaseConfig.bindTools([weatherTool]);
    const res = await chat.invoke([
      ['human', 'What\'s the weather in San Francisco?'],
    ]);
    const toolMessages = await Promise.all(res.tool_calls!.map(
      async (toolCall) =>
        new ToolMessage({
          tool_call_id: toolCall.id ?? 'testID',
          name: toolCall.name,
          content: await weatherTool.func({
            location: toolCall.args.location,
            unit: toolCall.args.unit,
          }),
        }),
    ));
    const finalResponse = await chat.invoke([
      ['human', 'What\'s the weather in San Francisco?'],
      res,
      ...toolMessages,
    ]);
    console.log(finalResponse);
    expect(finalResponse.text).toContain('72');
  }, 50000);

  test.skip('Test ChatOpenVINO tool calling with streaming', async () => {
    const chat = modelWithBaseConfig.bindTools(
      [
        {
          type: 'function',
          function: {
            name: 'get_current_weather',
            description: 'Get the current weather in a given location',
            parameters: {
              type: 'object',
              properties: {
                location: {
                  type: 'string',
                  description: 'The city and state, e.g. San Francisco, CA',
                },
                unit: { type: 'string', enum: ['celsius', 'fahrenheit'] },
              },
              required: ['location'],
            },
          },
        },
      ],
      {
        tool_choice: 'auto',
      },
    );
    const stream = await chat.stream([
      ['human', 'What\'s the weather like in San Francisco, Tokyo, and Paris?'],
    ]);
    let finalChunk: AIMessageChunk | undefined;
    const chunks: AIMessageChunk[] = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
      if (!finalChunk) {
        finalChunk = chunk;
      } else {
        finalChunk = finalChunk.concat(chunk);
      }
    }
    console.log(finalChunk);
    expect(chunks.length).toBeGreaterThan(1);
    expect(finalChunk?.tool_calls?.length).toBeGreaterThan(1);
    expect(finalChunk?.tool_calls?.[0].name).toBe('get_current_weather');
    expect(finalChunk?.tool_calls?.[0].args).toHaveProperty('location');
    expect(finalChunk?.tool_calls?.[0].id).toBeDefined();
  }, 50000);

  test.skip('Few shotting with tool calls', async () => {
    const chat = modelWithBaseConfig.bindTools([weatherTool]);
    const res = await chat.invoke([
      new HumanMessage('What is the weather in SF?'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: '12345',
            name: 'get_current_weather',
            args: {
              location: 'SF',
            },
          },
        ],
      }),
      new ToolMessage({
        tool_call_id: '12345',
        content: 'It is currently 24 degrees with hail in SF.',
      }),
      new AIMessage('It is currently 24 degrees in SF with hail in SF.'),
      new HumanMessage('What did you say the weather was?'),
    ]);
    expect(res.content).toContain('24');
  }, 50000);

  test.skip('Test tool calling with empty schema in streaming vs non-streaming',
    async () => {
    // Tool with empty schema (no parameters)
      const getCurrentTime = tool(
        async () => `current time: ${new Date().toLocaleString()}`,
        {
          name: 'get_current_time',
          description: 'get current time',
          schema: z.object({}),
        },
      );

      const llmWithTools = modelWithBaseConfig.bindTools([getCurrentTime]);

      const dialogs = [
        new SystemMessage({ content: 'You are a helpful assistant.' }),
        new HumanMessage({ content: 'get current time' }),
      ];

      // Test non-streaming mode - this should work
      const nonStreamingResult = await llmWithTools.invoke(dialogs);
      expect(nonStreamingResult.tool_calls).toBeDefined();
      expect(nonStreamingResult.tool_calls?.length).toBeGreaterThan(0);
      expect(nonStreamingResult.tool_calls?.[0].name).toBe('get_current_time');
      expect(nonStreamingResult.tool_calls?.[0].args).toEqual({});

      const stream = await llmWithTools.stream(dialogs);
      let finalChunk;
      for await (const chunk of stream) {
        if (!finalChunk) {
          finalChunk = chunk;
        } else {
          finalChunk = finalChunk.concat(chunk);
        }
      }

      expect(finalChunk?.tool_calls).toBeDefined();
      expect(finalChunk?.tool_calls?.length).toBeGreaterThan(0);
      expect(finalChunk?.tool_calls?.[0].name).toBe('get_current_time');
      expect(finalChunk?.tool_calls?.[0].args).toEqual({});
    }, 50000);

  test('Supports tool_choice', async () => {
    const tools = [
      {
        name: 'get_weather',
        description: 'Get the weather',
        schema: z.object({
          location: z.string(),
        }),
      },
      {
        name: 'calculator',
        description: 'Preform calculations',
        schema: z.object({
          expression: z.string(),
        }),
      },
    ];

    const modelWithTools = modelWithBaseConfig.bindTools(tools, {
      tool_choice: 'calculator',
    });
    const response = await modelWithTools.invoke(
      'What is 27725327 times 283683? Also whats the weather in New York?',
    );
    console.log(response);
    expect(response.tool_calls?.length).toBe(1);
  }, 100000);

});
