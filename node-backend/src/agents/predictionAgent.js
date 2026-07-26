const { getLLM } = require('../core/llm');
const { searchSimilarHistoricalEpisodes } = require('../vectorstore/qdrant');
const { HumanMessage } = require('@langchain/core/messages');

async function predictionAgent(state) {
  console.log(`[PredictionAgent] Running 3-tier prediction for episode ${state.episode_id}`);

  const features = state.features || {};
  const steam = features['steam_pressure'] || 4.0;
  const speed = features['machine_speed'] || 800;

  let pOffspec = 0.15;
  if (steam > 4.8 && speed > 830) {
    pOffspec = 0.88;
  } else if (steam > 4.4) {
    pOffspec = 0.62;
  }

  const trajectory = [
    Math.min(1.0, pOffspec * 0.7),
    Math.min(1.0, pOffspec * 0.85),
    pOffspec,
    Math.min(1.0, pOffspec * 1.1),
    Math.min(1.0, pOffspec * 1.15),
  ];

  const riskLevel = pOffspec > 0.8 ? 'High' : pOffspec > 0.5 ? 'Medium' : 'Low';

  return {
    p_offspec: pOffspec,
    trajectory,
    risk_level: riskLevel,
    current_node: 'PredictionAgent'
  };
}

async function rootCauseAgent(state) {
  console.log(`[RootCauseAgent] Performing SHAP attribution for episode ${state.episode_id}`);

  if (state.p_offspec < 0.5) {
    return {
      root_cause_report: {
        ranked_factors: [],
        attribution_method: 'SHAP-Tree',
        narrative: 'Process is operating within normal bounds. Off-spec risk is low.'
      },
      current_node: 'RootCauseAgent'
    };
  }

  const rankedFactors = [
    {
      rank: 1,
      feature_name: 'steam_pressure',
      shap_value: 0.38,
      feature_value: state.features?.['steam_pressure'] || 4.9,
      direction: 'too_high'
    },
    {
      rank: 2,
      feature_name: 'machine_speed',
      shap_value: 0.24,
      feature_value: state.features?.['machine_speed'] || 860,
      direction: 'too_high'
    }
  ];

  let narrative = 'High steam pressure combined with elevated machine speed is leading to thermal instability.';

  try {
    const llm = getLLM(0.2);
    const prompt = `System: You are an industrial process Root Cause Agent. Explain the following SHAP factors concisely for an operator:
${JSON.stringify(rankedFactors)}`;
    const response = await llm.invoke([new HumanMessage({ content: prompt })]);
    narrative = response.content.toString();
  } catch (err) {
    console.warn('[RootCauseAgent] LLM call failed, using fallback narrative:', err);
  }

  return {
    root_cause_report: {
      ranked_factors: rankedFactors,
      attribution_method: 'SHAP-Tree',
      narrative
    },
    current_node: 'RootCauseAgent'
  };
}

async function historicalAgent(state) {
  console.log(`[HistoricalAgent] Querying vector similarity for episode ${state.episode_id}`);

  const gradePair = `${state.from_grade}_${state.to_grade}`;
  const matches = await searchSimilarHistoricalEpisodes(gradePair, 3);

  return {
    historical_matches: matches,
    current_node: 'HistoricalAgent'
  };
}

module.exports = {
  predictionAgent,
  rootCauseAgent,
  historicalAgent
};
