const { Router } = require('express');
const { Recommendation, Feedback } = require('../models');

const router = Router();

router.get('/episode/:episodeId', async (req, res) => {
  try {
    const recs = await Recommendation.find({ episode_id: req.params.episodeId }).sort({ createdAt: -1 });
    res.json(recs);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/:id/feedback', async (req, res) => {
  try {
    const { operator_id, action, comment } = req.body;
    const recId = req.params.id;

    const feedback = new Feedback({
      feedback_id: `fb-${Date.now()}`,
      recommendation_id: recId,
      operator_id: operator_id || 'op-default',
      action,
      comment
    });
    await feedback.save();

    await Recommendation.updateOne(
      { recommendation_id: recId },
      { $set: { operator_action: action } }
    );

    res.json({ message: 'Feedback recorded', feedback });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
