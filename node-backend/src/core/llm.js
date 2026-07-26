const { ChatGroq } = require('@langchain/groq');
const { config } = require('./config');

function getLLM(temperature = 0.2) {
  const apiKey = config.groqApiKey || process.env.GROQ_API_KEY;
  if (!apiKey) {
    console.warn('[LLM] GROQ_API_KEY is missing in environment!');
  }
  return new ChatGroq({
    apiKey: apiKey || 'dummy-key',
    modelName: 'llama-3.3-70b-versatile',
    temperature: temperature,
  });
}

module.exports = { getLLM };
