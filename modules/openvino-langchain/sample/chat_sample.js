import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { ChatOpenVINO } from 'openvino-langchain';
import { basename } from 'node:path';
import readline from 'readline';

const LLM_MODEL_PATH = process.argv[2];

if (!LLM_MODEL_PATH) {
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

async function main() {
  const MODEL_PATH = process.argv[2];

  if (process.argv.length > 3) {
    console.error(
      `Run command must be:
      'node ${basename(process.argv[1])} *path_to_model_dir*'`,
    );
    process.exit(1);
  }
  if (!MODEL_PATH) {
    console.error('Please specify path to model directory\n'
      + `Run command must be:
      'node ${basename(process.argv[1])} *path_to_model_dir*'`);
    process.exit(1);
  }

  const device = 'CPU'; // GPU can be used as well
  const config = { 'max_new_tokens': 100 };
  const chat = new ChatOpenVINO({
    modelPath: LLM_MODEL_PATH,
    device,
    generationConfig: config,
  });

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const messages = [
    new SystemMessage('You are chatbot.'),
  ];

  promptUser();

  // Function to prompt the user for input
  function promptUser() {
    rl.question('question:\n', handleInput);
  }

  // Function to handle user input
  async function handleInput(input) {
    input = input.trim();

    // Check for exit command
    if (!input) {
      rl.close();
      process.exit(0);
    }

    messages.push(new HumanMessage(input));
    const aiResponse = await chat.invoke(messages);

    messages.push(aiResponse);
    console.log(aiResponse.text);
    console.log('\n----------');

    promptUser();
  }
}

main();
