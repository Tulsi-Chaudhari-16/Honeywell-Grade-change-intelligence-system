const { StateGraph, Annotation, END } = require('@langchain/langgraph');
const { dataAgent, featureAgent } = require('./dataAgent');
const { predictionAgent, rootCauseAgent, historicalAgent } = require('./predictionAgent');
const { recommendationAgent, safetyValidationAgent, explanationAgent } = require('./recommendationAgent');
const { alertAgent, dashboardAgent } = require('./supportingAgents');

const EpisodeAnnotation = Annotation.Root({
  episode_id: Annotation({ value: (x, y) => y ?? x, default: () => '' }),
  machine_id: Annotation({ value: (x, y) => y ?? x, default: () => 'PM1' }),
  from_grade: Annotation({ value: (x, y) => y ?? x, default: () => 'GRADE-A' }),
  to_grade: Annotation({ value: (x, y) => y ?? x, default: () => 'GRADE-B' }),
  raw_tags: Annotation({ value: (x, y) => y ?? x }),
  features: Annotation({ value: (x, y) => y ?? x }),
  p_offspec: Annotation({ value: (x, y) => y ?? x, default: () => 0.0 }),
  trajectory: Annotation({ value: (x, y) => y ?? x, default: () => [] }),
  risk_level: Annotation({ value: (x, y) => y ?? x, default: () => 'Low' }),
  root_cause_report: Annotation({ value: (x, y) => y ?? x }),
  historical_matches: Annotation({ value: (x, y) => y ?? x }),
  candidate_recommendations: Annotation({ value: (x, y) => y ?? x }),
  validated_candidates: Annotation({ value: (x, y) => y ?? x }),
  explanation_card: Annotation({ value: (x, y) => y ?? x }),
  operator_action: Annotation({ value: (x, y) => y ?? x }),
  current_node: Annotation({ value: (x, y) => y ?? x }),
  error: Annotation({ value: (x, y) => y ?? x })
});

function buildSupervisorGraph() {
  const workflow = new StateGraph(EpisodeAnnotation)
    .addNode('data_agent', dataAgent)
    .addNode('feature_agent', featureAgent)
    .addNode('prediction_agent', predictionAgent)
    .addNode('root_cause_agent', rootCauseAgent)
    .addNode('historical_agent', historicalAgent)
    .addNode('recommendation_agent', recommendationAgent)
    .addNode('safety_validation_agent', safetyValidationAgent)
    .addNode('explanation_agent', explanationAgent)
    .addNode('alert_agent', alertAgent)
    .addNode('dashboard_agent', dashboardAgent)

    .addEdge('__start__', 'data_agent')
    .addEdge('data_agent', 'feature_agent')
    .addEdge('feature_agent', 'prediction_agent')
    .addEdge('prediction_agent', 'root_cause_agent')
    .addEdge('root_cause_agent', 'historical_agent')
    .addEdge('historical_agent', 'recommendation_agent')
    .addEdge('recommendation_agent', 'safety_validation_agent')
    .addEdge('safety_validation_agent', 'explanation_agent')
    .addEdge('explanation_agent', 'alert_agent')
    .addEdge('alert_agent', 'dashboard_agent')
    .addEdge('dashboard_agent', END);

  return workflow.compile();
}

const supervisorGraph = buildSupervisorGraph();

module.exports = {
  EpisodeAnnotation,
  buildSupervisorGraph,
  supervisorGraph
};
