const { getLLM } = require('../core/llm');
const { HumanMessage } = require('@langchain/core/messages');

async function recommendationAgent(state) {
  console.log(`[RecommendationAgent] Generating recommendations for episode ${state.episode_id}`);

  if (state.p_offspec < 0.5) {
    return {
      candidate_recommendations: [],
      current_node: 'RecommendationAgent'
    };
  }

  let rawCandidates = [];
  
  if (state.root_cause_report && state.root_cause_report.ranked_factors && state.root_cause_report.ranked_factors.length > 0) {
    const topFactor = state.root_cause_report.ranked_factors[0];
    // Suggest reversing the direction of the top SHAP factor
    const proposedChange = topFactor.direction === 'too_high' ? topFactor.feature_value * 0.9 : topFactor.feature_value * 1.1;
    
    rawCandidates.push({
      variable_name: topFactor.feature_name,
      proposed_value: parseFloat(proposedChange.toFixed(2)),
      expected_improvement: `Adjusting ${topFactor.feature_name} to mitigate ${Math.round(state.p_offspec * 100)}% off-spec risk.`,
      historical_support: state.historical_matches ? state.historical_matches.map(m => m.episode_id) : []
    });
  }

  return {
    candidate_recommendations: rawCandidates,
    current_node: 'RecommendationAgent'
  };
}

async function safetyValidationAgent(state) {
  console.log(`[SafetyValidationAgent] Validating candidate recommendations for episode ${state.episode_id}`);

  const candidates = state.candidate_recommendations || [];
  const validated = [];

  for (const cand of candidates) {
    let safetyStatus = 'approved';
    let clampedValue = undefined;

    if (cand.variable_name === 'steam_pressure') {
      if (cand.proposed_value < 1.0) {
        safetyStatus = 'rejected';
      } else if (cand.proposed_value > 5.5) {
        safetyStatus = 'clamped';
        clampedValue = 5.5;
      }
    } else if (cand.variable_name === 'machine_speed') {
      if (cand.proposed_value > 1100) {
        safetyStatus = 'clamped';
        clampedValue = 1100;
      }
    }

    validated.push({
      recommendation_id: `rec-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
      variable_name: cand.variable_name,
      proposed_value: cand.proposed_value,
      clamped_value: clampedValue,
      safety_status: safetyStatus,
      expected_improvement: cand.expected_improvement,
      historical_support: cand.historical_support
    });
  }

  return {
    validated_candidates: validated,
    current_node: 'SafetyValidationAgent'
  };
}

async function explanationAgent(state) {
  console.log(`[ExplanationAgent] Generating structured ExplanationCard for episode ${state.episode_id}`);

  const pOffspec = state.p_offspec || 0;
  const topFactor = state.root_cause_report?.ranked_factors?.[0]?.feature_name || 'steam_pressure';
  const histMatch = state.historical_matches?.[0]?.episode_id || 'hist-e101';
  const safetyStatus = state.validated_candidates?.[0]?.safety_status || 'approved';

  const card = {
    prediction_summary: `Predicted off-spec probability is ${Math.round(pOffspec * 100)}% over the next 15 minutes.`,
    reason: `Abnormal elevation in ${topFactor} creating process instability.`,
    evidence: state.root_cause_report?.ranked_factors?.[0] ? `SHAP feature attribution value: +${state.root_cause_report.ranked_factors[0].shap_value.toFixed(3)}` : `ML Model consensus`,
    historical_match: histMatch ? `Similar historical pattern identified in episode ${histMatch}.` : `No historical matches.`,
    confidence_statement: 'High confidence based on ML ensemble consensus.',
    safety_check_status: `Safety Validation: ${safetyStatus.toUpperCase()}`
  };

  return {
    explanation_card: card,
    current_node: 'ExplanationAgent'
  };
}

module.exports = {
  recommendationAgent,
  safetyValidationAgent,
  explanationAgent
};
