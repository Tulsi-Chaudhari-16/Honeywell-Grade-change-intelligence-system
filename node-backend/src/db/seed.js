require('dotenv').config();
const { connectDB } = require('./connect');
const { Recipe, MachineLimit } = require('../models');

async function seed() {
  await connectDB();
  console.log('[Seed] Seeding default recipes and machine limits...');

  await Recipe.deleteMany({});
  await MachineLimit.deleteMany({});

  await Recipe.insertMany([
    {
      code: 'GRADE-A',
      name: 'Standard Kraft 80gsm',
      targets: { basis_weight: 80.0, moisture: 6.5, machine_speed: 850.0 },
      limits: {
        machine_speed: { min: 700.0, max: 1000.0 },
        steam_pressure: { min: 2.0, max: 6.0 },
        stock_flow_rate: { min: 3000.0, max: 5000.0 },
        headbox_pressure: { min: 1.0, max: 3.0 }
      }
    },
    {
      code: 'GRADE-B',
      name: 'Heavy Linerboard 120gsm',
      targets: { basis_weight: 120.0, moisture: 7.0, machine_speed: 720.0 },
      limits: {
        machine_speed: { min: 600.0, max: 900.0 },
        steam_pressure: { min: 3.0, max: 7.0 },
        stock_flow_rate: { min: 4000.0, max: 6500.0 },
        headbox_pressure: { min: 1.5, max: 3.5 }
      }
    },
    {
      code: 'GRADE-C',
      name: 'Ultra Lightweight 60gsm',
      targets: { basis_weight: 60.0, moisture: 5.5, machine_speed: 980.0 },
      limits: {
        machine_speed: { min: 800.0, max: 1100.0 },
        steam_pressure: { min: 1.5, max: 5.0 },
        stock_flow_rate: { min: 2500.0, max: 4200.0 },
        headbox_pressure: { min: 0.8, max: 2.5 }
      }
    }
  ]);

  await MachineLimit.insertMany([
    { machine_id: 'PM1', tag_name: 'machine_speed', min_val: 500.0, max_val: 1200.0, max_rate_of_change: 15.0 },
    { machine_id: 'PM1', tag_name: 'steam_pressure', min_val: 1.0, max_val: 8.0, max_rate_of_change: 0.5 },
    { machine_id: 'PM1', tag_name: 'stock_flow_rate', min_val: 2000.0, max_val: 7000.0, max_rate_of_change: 100.0 },
    { machine_id: 'PM1', tag_name: 'headbox_pressure', min_val: 0.5, max_val: 4.0, max_rate_of_change: 0.2 }
  ]);

  console.log('[Seed] Database seeded successfully!');
  process.exit(0);
}

seed().catch(err => {
  console.error('[Seed] Error seeding database:', err);
  process.exit(1);
});
