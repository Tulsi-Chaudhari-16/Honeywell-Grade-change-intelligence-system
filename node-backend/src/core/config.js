require('dotenv').config();

const config = {
  port: parseInt(process.env.PORT || '8000', 10),
  mongoUri: process.env.MONGO_URI || 'mongodb://localhost:27017/gcis_db',
  redisHost: process.env.REDIS_HOST || 'localhost',
  redisPort: parseInt(process.env.REDIS_PORT || '6379', 10),
  qdrantHost: process.env.QDRANT_HOST || 'localhost',
  qdrantPort: parseInt(process.env.QDRANT_PORT || '6333', 10),
  kafkaBootstrapServers: process.env.KAFKA_BOOTSTRAP_SERVERS || 'localhost:9092',
  groqApiKey: process.env.GROQ_API_KEY || '',
};

module.exports = { config };
