const { Alert, Transition, Recommendation } = require('../models');
const { redisPub } = require('../core/redis');

async function operatorFeedbackAgent(state, recId, action) {
  console.log(`[OperatorFeedbackAgent] Recording operator action '${action}' for rec ${recId}`);

  await Recommendation.updateOne(
    { recommendation_id: recId },
    { $set: { operator_action: action } }
  );

  return {
    operator_action: action,
    current_node: 'OperatorFeedbackAgent'
  };
}

async function alertAgent(state) {
  console.log(`[AlertAgent] Evaluating alerts for episode ${state.episode_id}`);

  if (state.p_offspec > 0.8) {
    const alertId = `alert-${Date.now()}`;
    const alertDoc = new Alert({
      alert_id: alertId,
      episode_id: state.episode_id,
      alert_type: 'HIGH_OFFSPEC_RISK',
      severity: 'critical',
      message: `Off-spec probability reached ${Math.round(state.p_offspec * 100)}% on ${state.machine_id}`,
      acknowledged: false
    });
    await alertDoc.save();

    try {
      await redisPub.publish('gcis:alerts', JSON.stringify(alertDoc.toJSON()));
    } catch (e) {}
  }

  return {
    current_node: 'AlertAgent'
  };
}

async function learningAgent(state) {
  console.log(`[LearningAgent] Logging post-episode telemetry for episode ${state.episode_id}`);

  await Transition.updateOne(
    { episode_id: state.episode_id },
    {
      $set: {
        status: 'Resolved',
        end_time: new Date(),
        p_offspec: state.p_offspec
      }
    }
  );

  return {
    current_node: 'LearningAgent'
  };
}

async function dashboardAgent(state) {
  console.log(`[DashboardAgent] Broadcasting websocket telemetry for episode ${state.episode_id}`);

  try {
    await redisPub.publish(`episode:${state.episode_id}`, JSON.stringify(state));
  } catch (e) {}

  return {
    current_node: 'DashboardAgent'
  };
}

module.exports = {
  operatorFeedbackAgent,
  alertAgent,
  learningAgent,
  dashboardAgent
};
