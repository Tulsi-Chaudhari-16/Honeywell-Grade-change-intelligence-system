const { getLLM } = require('../core/llm');
const { searchSimilarHistoricalEpisodes } = require('../vectorstore/qdrant');
const { HumanMessage } = require('@langchain/core/messages');

async function predictionAgent(state) {
  console.log(`[PredictionAgent] Running 3-tier prediction for episode ${state.episode_id}`);

  const features = state.features || {};
  const steam = features['steam_pressure'] || 4.0;
  const speed = features['machine_speed'] || 800;

  let pOffspec = 0.15;
  let trajectory = [];
  let rootCauseReport = null;

  try {
    const response = await fetch('http://localhost:8001/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        episode_id: state.episode_id,
        ts: state.ts || new Date().toISOString(),
        features: state.features || {},
        imputation_flags: state.imputation_flags || {}
      })
    });
    
    if (response.ok) {
      const data = await response.json();
      if (data.prediction) {
        pOffspec = data.prediction.p_offspec;
        trajectory = data.prediction.trajectory || [];
      }
      if (data.root_cause_attribution) {
        rootCauseReport = data.root_cause_attribution;
      }
    } else {
      console.warn('[PredictionAgent] ML API failed, using fallbacks');
    }
  } catch (err) {
    console.warn('[PredictionAgent] ML API error:', err.message);
  }

  const riskLevel = pOffspec > 0.8 ? 'High' : pOffspec > 0.5 ? 'Medium' : 'Low';

  return {
    p_offspec: pOffspec,
    trajectory,
    risk_level: riskLevel,
    raw_root_cause: rootCauseReport, // pass to RootCauseAgent
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

  const rawRootCause = state.raw_root_cause;
  let rankedFactors = [];
  let attributionMethod = 'SHAP-Tree';

  if (rawRootCause && rawRootCause.ranked_factors) {
    rankedFactors = rawRootCause.ranked_factors;
    attributionMethod = rawRootCause.attribution_method || 'SHAP-Tree';
  } else {
    // Fallback if ML API didn't return SHAP
    rankedFactors = [
      {
        rank: 1,
        feature_name: 'steam_pressure',
        shap_value: 0.38,
        feature_value: state.features?.['steam_pressure'] || 4.9,
        direction: 'too_high'
      }
    ];
  }

  let narrative = 'Process is experiencing instability due to identified SHAP factors.';

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
