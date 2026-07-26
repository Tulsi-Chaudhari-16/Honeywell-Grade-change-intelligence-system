const mongoose = require('mongoose');
const { Schema } = mongoose;

// 1. Transition / Episode Schema
const TransitionSchema = new Schema({
  episode_id: { type: String, required: true, unique: true, index: true },
  machine_id: { type: String, required: true },
  from_grade: { type: String, required: true },
  to_grade: { type: String, required: true },
  start_time: { type: Date, default: Date.now },
  end_time: { type: Date },
  status: { type: String, default: 'FeatureBuilding' },
  final_outcome: { type: String },
  p_offspec: { type: Number, default: 0.0 }
}, { timestamps: true });

const Transition = mongoose.model('Transition', TransitionSchema);

// 2. Prediction Schema
const PredictionSchema = new Schema({
  prediction_id: { type: String, required: true, unique: true },
  episode_id: { type: String, required: true, index: true },
  timestamp: { type: Date, default: Date.now },
  p_offspec: { type: Number, required: true },
  trajectory: [{ type: Number }],
  risk_level: { type: String, required: true },
  model_version: { type: String, default: '1.0.0-js' }
}, { timestamps: true });

const Prediction = mongoose.model('Prediction', PredictionSchema);

// 3. Recommendation Schema
const RecommendationSchema = new Schema({
  recommendation_id: { type: String, required: true, unique: true },
  episode_id: { type: String, required: true, index: true },
  timestamp: { type: Date, default: Date.now },
  variable_name: { type: String, required: true },
  proposed_value: { type: Number, required: true },
  clamped_value: { type: Number },
  safety_status: { type: String, required: true },
  expected_improvement: { type: String, required: true },
  historical_support: [{ type: String }],
  operator_action: { type: String }
}, { timestamps: true });

const Recommendation = mongoose.model('Recommendation', RecommendationSchema);

// 4. Alert Schema
const AlertSchema = new Schema({
  alert_id: { type: String, required: true, unique: true },
  episode_id: { type: String },
  timestamp: { type: Date, default: Date.now },
  alert_type: { type: String, required: true },
  severity: { type: String, required: true },
  message: { type: String, required: true },
  acknowledged: { type: Boolean, default: false }
}, { timestamps: true });

const Alert = mongoose.model('Alert', AlertSchema);

// 5. Operator Feedback Schema
const FeedbackSchema = new Schema({
  feedback_id: { type: String, required: true, unique: true },
  recommendation_id: { type: String, required: true, index: true },
  operator_id: { type: String, required: true },
  timestamp: { type: Date, default: Date.now },
  action: { type: String, required: true },
  comment: { type: String }
}, { timestamps: true });

const Feedback = mongoose.model('Feedback', FeedbackSchema);

// 6. Recipe Schema
const RecipeSchema = new Schema({
  code: { type: String, required: true, unique: true },
  name: { type: String, required: true },
  targets: { type: Schema.Types.Mixed, default: {} },
  limits: { type: Schema.Types.Mixed, default: {} }
}, { timestamps: true });

const Recipe = mongoose.model('Recipe', RecipeSchema);

// 7. Machine Limit Schema
const MachineLimitSchema = new Schema({
  machine_id: { type: String, required: true },
  tag_name: { type: String, required: true },
  min_val: { type: Number, required: true },
  max_val: { type: Number, required: true },
  max_rate_of_change: { type: Number, required: true }
}, { timestamps: true });

MachineLimitSchema.index({ machine_id: 1, tag_name: 1 }, { unique: true });

const MachineLimit = mongoose.model('MachineLimit', MachineLimitSchema);

module.exports = {
  Transition,
  Prediction,
  Recommendation,
  Alert,
  Feedback,
  Recipe,
  MachineLimit
};
