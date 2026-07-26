const { WebSocketServer, WebSocket } = require('ws');
const { redisSub } = require('../core/redis');

function setupWebSockets(server) {
  const wss = new WebSocketServer({ server });

  wss.on('connection', (ws, req) => {
    const url = req.url || '';
    console.log(`[WebSocket] Client connected: ${url}`);

    if (url.startsWith('/ws/episodes/')) {
      const episodeId = url.split('/ws/episodes/')[1];
      console.log(`[WebSocket] Subscribing to episode ${episodeId}`);

      redisSub.subscribe(`episode:${episodeId}`, (err) => {
        if (err) console.error('[WebSocket] Redis subscribe error:', err);
      });

      const messageHandler = (channel, message) => {
        if (channel === `episode:${episodeId}` && ws.readyState === WebSocket.OPEN) {
          ws.send(message);
        }
      };

      redisSub.on('message', messageHandler);

      ws.on('close', () => {
        redisSub.off('message', messageHandler);
        redisSub.unsubscribe(`episode:${episodeId}`);
        console.log(`[WebSocket] Client disconnected from episode ${episodeId}`);
      });

      ws.send(JSON.stringify({
        episode_id: episodeId,
        p_offspec: 0.12,
        trajectory: [0.1, 0.12, 0.15],
        current_node: 'Initialized'
      }));

    } else if (url === '/ws/alerts') {
      console.log('[WebSocket] Subscribing to system alerts');

      redisSub.subscribe('gcis:alerts');

      const alertHandler = (channel, message) => {
        if (channel === 'gcis:alerts' && ws.readyState === WebSocket.OPEN) {
          ws.send(message);
        }
      };

      redisSub.on('message', alertHandler);

      ws.on('close', () => {
        redisSub.off('message', alertHandler);
        redisSub.unsubscribe('gcis:alerts');
      });
    }
  });

  console.log('[WebSocket] WebSocket server initialized');
}

module.exports = { setupWebSockets };
