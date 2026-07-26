const { Router } = require('express');
const { Transition } = require('../models');
const { supervisorGraph } = require('../agents/supervisor');

const router = Router();

router.get('/active', async (req, res) => {
  try {
    const active = await Transition.find({ status: { $ne: 'Resolved' } }).sort({ createdAt: -1 });
    res.json(active);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.get('/:id', async (req, res) => {
  try {
    const ep = await Transition.findOne({ episode_id: req.params.id });
    if (!ep) return res.status(404).json({ error: 'Episode not found' });
    res.json(ep);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/start', async (req, res) => {
  try {
    const { machine_id, from_grade, to_grade } = req.body;
    const episodeId = `ep-${Date.now()}`;

    const transition = new Transition({
      episode_id: episodeId,
      machine_id: machine_id || 'PM1',
      from_grade: from_grade || 'GRADE-A',
      to_grade: to_grade || 'GRADE-B',
      status: 'FeatureBuilding',
      p_offspec: 0.12
    });
    await transition.save();

    supervisorGraph.invoke({
      episode_id: episodeId,
      machine_id: transition.machine_id,
      from_grade: transition.from_grade,
      to_grade: transition.to_grade,
      p_offspec: 0.12,
      trajectory: [0.1, 0.12, 0.15],
      risk_level: 'Low'
    }).catch(err => console.error('[SupervisorGraph] Execution error:', err));

    res.status(201).json(transition);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
