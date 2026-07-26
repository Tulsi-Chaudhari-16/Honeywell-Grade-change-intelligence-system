const Redis = require('ioredis');
const { config } = require('./config');

const redis = new Redis({
  host: config.redisHost,
  port: config.redisPort,
  lazyConnect: true,
});

const redisPub = new Redis({
  host: config.redisHost,
  port: config.redisPort,
  lazyConnect: true,
});

const redisSub = new Redis({
  host: config.redisHost,
  port: config.redisPort,
  lazyConnect: true,
});

async function initRedis() {
  try {
    await redis.connect();
    await redisPub.connect();
    await redisSub.connect();
    console.log('[Redis] Connected successfully');
  } catch (err) {
    console.warn('[Redis] Connection failed or running in fallback mode:', err);
  }
}

module.exports = {
  redis,
  redisPub,
  redisSub,
  initRedis
};
