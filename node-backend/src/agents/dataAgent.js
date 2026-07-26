const { redis } = require('../core/redis');

async function dataAgent(state) {
  console.log(`[DataAgent] Processing raw tags for episode ${state.episode_id}`);

  let rawTags = state.raw_tags;
  if (!rawTags) {
    try {
      const cached = await redis.get(`lkg:${state.machine_id}`);
      if (cached) {
        rawTags = JSON.parse(cached);
      }
    } catch (e) {
      console.warn('[DataAgent] Redis cache read failed, using defaults');
    }
  }

  if (!rawTags) {
    rawTags = {
      machine_speed: 850.0,
      steam_pressure: 4.5,
      stock_flow_rate: 3800.0,
      headbox_pressure: 2.1,
      basis_weight: 81.2,
      moisture: 6.4,
    };
  }

  try {
    await redis.set(`lkg:${state.machine_id}`, JSON.stringify(rawTags));
  } catch (e) {}

  return {
    raw_tags: rawTags,
    current_node: 'DataAgent'
  };
}

async function featureAgent(state) {
  console.log(`[FeatureAgent] Calculating features for episode ${state.episode_id}`);

  const raw = state.raw_tags || {};
  const features = { ...raw };

  features['speed_to_steam_ratio'] = raw.steam_pressure ? raw.machine_speed / raw.steam_pressure : 0;
  features['flow_to_headbox_ratio'] = raw.headbox_pressure ? raw.stock_flow_rate / raw.headbox_pressure : 0;
  features['bw_moisture_product'] = (raw.basis_weight || 0) * (raw.moisture || 0);

  return {
    features,
    current_node: 'FeatureAgent'
  };
}

module.exports = {
  dataAgent,
  featureAgent
};
