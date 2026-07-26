const { QdrantClient } = require('@qdrant/js-client-rest');
const { config } = require('../core/config');

const qdrant = new QdrantClient({
  url: `http://${config.qdrantHost}:${config.qdrantPort}`,
});

async function searchSimilarHistoricalEpisodes(gradePair, limit = 3) {
  try {
    const collections = await qdrant.getCollections();
    const exists = collections.collections.some(c => c.name === 'gcis_episodes');

    if (exists) {
      const results = await qdrant.search('gcis_episodes', {
        vector: new Array(15).fill(0.1),
        limit,
        filter: {
          must: [{ key: 'grade_pair', match: { value: gradePair } }]
        }
      });
      return results.map(r => r.payload);
    }
  } catch (err) {
    console.warn('[Qdrant] Search failed or Qdrant unavailable, falling back to mock similarity:', err);
  }

  return [
    {
      episode_id: 'hist-e101',
      grade_pair: gradePair,
      outcome: 'success',
      recovery_time_min: 28,
      actions_taken: 'Decreased steam_pressure by 0.3 bar, increased machine_speed by 1.5%',
    },
    {
      episode_id: 'hist-e102',
      grade_pair: gradePair,
      outcome: 'success',
      recovery_time_min: 35,
      actions_taken: 'Adjusted stock_flow_rate to 4200 L/min',
    },
    {
      episode_id: 'hist-e103',
      grade_pair: gradePair,
      outcome: 'offspec',
      recovery_time_min: 90,
      actions_taken: 'Excessive speed change caused sheet break',
    }
  ];
}

module.exports = {
  qdrant,
  searchSimilarHistoricalEpisodes
};
