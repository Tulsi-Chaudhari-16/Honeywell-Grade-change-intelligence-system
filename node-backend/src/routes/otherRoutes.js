const { Router } = require('express');
const { Recipe, Alert } = require('../models');
const { supervisorGraph } = require('../agents/supervisor');
const { getLLM } = require('../core/llm');
const { HumanMessage, SystemMessage } = require('@langchain/core/messages');

// 1. Recipes Router
const recipesRouter = Router();
recipesRouter.get('/', async (req, res) => {
  try {
    const recipes = await Recipe.find({});
    res.json(recipes);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// 2. Alerts Router
const alertsRouter = Router();
alertsRouter.get('/', async (req, res) => {
  try {
    const alerts = await Alert.find({ acknowledged: false }).sort({ createdAt: -1 });
    res.json(alerts);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

alertsRouter.post('/:id/acknowledge', async (req, res) => {
  try {
    await Alert.updateOne({ alert_id: req.params.id }, { $set: { acknowledged: true } });
    res.json({ message: 'Alert acknowledged' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// 3. Simulate Router
const simulateRouter = Router();
simulateRouter.post('/', async (req, res) => {
  try {
    const { machine_id, from_grade, to_grade, scenario } = req.body;
    const episodeId = `sim-${Date.now()}`;

    const finalState = await supervisorGraph.invoke({
      episode_id: episodeId,
      machine_id: machine_id || 'PM1',
      from_grade: from_grade || 'GRADE-A',
      to_grade: to_grade || 'GRADE-B',
      p_offspec: scenario === 'high_risk' ? 0.88 : 0.15,
      trajectory: [0.2, 0.4, 0.88],
      risk_level: scenario === 'high_risk' ? 'High' : 'Low'
    });

    res.json({ message: 'Simulation executed', result: finalState });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// 4. Chat Router
const chatRouter = Router();
chatRouter.post('/', async (req, res) => {
  try {
    const { message } = req.body;
    const llm = getLLM(0.7);
    const response = await llm.invoke([
      new SystemMessage({ content: 'You are GCIS (Grade Change Intelligence System). Assist the operator concisely.' }),
      new HumanMessage({ content: message })
    ]);
    res.json({ reply: response.content });
  } catch (err) {
    res.json({ reply: `Error: ${err.message}` });
  }
});

// 5. System Router
const systemRouter = Router();
systemRouter.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    backend: 'Express & Node.js (JavaScript MERN)',
    provider: 'Groq (llama-3.3-70b-versatile)'
  });
});

module.exports = {
  recipesRouter,
  alertsRouter,
  simulateRouter,
  chatRouter,
  systemRouter
};
