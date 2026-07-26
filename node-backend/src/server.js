const http = require('http');
const express = require('express');
const cors = require('cors');
const { config } = require('./core/config');
const { connectDB } = require('./db/connect');
const { initRedis } = require('./core/redis');
const episodesRouter = require('./routes/episodes');
const recommendationsRouter = require('./routes/recommendations');
const { recipesRouter, alertsRouter, simulateRouter, chatRouter, systemRouter } = require('./routes/otherRoutes');
const { setupWebSockets } = require('./websockets');

async function bootstrap() {
  // 1. Connect to Database & Redis
  await connectDB(config.mongoUri);
  await initRedis();

  // 2. Express App Setup
  const app = express();
  app.use(cors());
  app.use(express.json());

  // 3. Register REST API Routers
  app.use('/api/v1/episodes', episodesRouter);
  app.use('/api/v1/recommendations', recommendationsRouter);
  app.use('/api/v1/recipes', recipesRouter);
  app.use('/api/v1/alerts', alertsRouter);
  app.use('/api/v1/simulate', simulateRouter);
  app.use('/api/v1/chat', chatRouter);
  app.use('/api/v1/system', systemRouter);

  // 4. HTTP & WebSocket Server Creation
  const server = http.createServer(app);
  setupWebSockets(server);

  server.listen(config.port, () => {
    console.log(`=================================================`);
    console.log(`  GCIS MERN Backend running on port ${config.port}`);
    console.log(`  MongoDB: ${config.mongoUri}`);
    console.log(`  LLM Provider: Groq (llama-3.3-70b-versatile)`);
    console.log(`=================================================`);
  });
}

bootstrap().catch(err => {
  console.error('Fatal server startup error:', err);
  process.exit(1);
});
