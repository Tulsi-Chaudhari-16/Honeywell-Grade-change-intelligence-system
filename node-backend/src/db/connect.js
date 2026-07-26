const mongoose = require('mongoose');

async function connectDB(uri) {
  const mongoUri = uri || process.env.MONGO_URI || 'mongodb://localhost:27017/gcis_db';
  try {
    await mongoose.connect(mongoUri);
    console.log(`[MongoDB] Connected successfully to ${mongoUri}`);
  } catch (error) {
    console.error('[MongoDB] Connection error:', error);
    process.exit(1);
  }
}

module.exports = { connectDB };
