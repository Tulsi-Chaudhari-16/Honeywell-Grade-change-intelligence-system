-- ============================================================
-- GCIS Seed Data — Migration 002
-- Demo recipes, machine limits, and operators
-- ============================================================

-- ─── Demo Operators ───────────────────────────────────────────────────────────
INSERT INTO operators (operator_id, display_name, shift) VALUES
    ('a0000000-0000-0000-0000-000000000001', 'Alice Sharma',   'Day'),
    ('a0000000-0000-0000-0000-000000000002', 'Bob Mendes',     'Night'),
    ('a0000000-0000-0000-0000-000000000003', 'Carol Zhang',    'Day'),
    ('a0000000-0000-0000-0000-000000000004', 'Demo Operator',  'Day')
ON CONFLICT DO NOTHING;

-- ─── Grade Recipes ────────────────────────────────────────────────────────────
INSERT INTO recipes (recipe_id, grade_code, grade_name, target_basis_weight, target_moisture, target_ash, target_caliper) VALUES
    ('b0000000-0000-0000-0000-000000000001', 'G70',  'Newsprint 70gsm',         70.0,  8.5, 12.0, 0.085),
    ('b0000000-0000-0000-0000-000000000002', 'G80',  'Standard 80gsm',          80.0,  8.0, 10.0, 0.095),
    ('b0000000-0000-0000-0000-000000000003', 'G90',  'Premium 90gsm',           90.0,  7.5,  8.0, 0.105),
    ('b0000000-0000-0000-0000-000000000004', 'G100', 'Heavy 100gsm',           100.0,  7.0,  6.0, 0.120),
    ('b0000000-0000-0000-0000-000000000005', 'G55',  'Lightweight 55gsm',       55.0,  9.5, 15.0, 0.065),
    ('b0000000-0000-0000-0000-000000000006', 'G120', 'Coated Board 120gsm',    120.0,  6.0,  5.0, 0.140)
ON CONFLICT DO NOTHING;

-- ─── Machine Limits (Paper Machine PM1) ──────────────────────────────────────
INSERT INTO machine_limits (machine_id, variable_name, min_value, max_value, max_rate_of_change, unit) VALUES
    -- Speed
    ('PM1', 'machine_speed',          400,   1200,  50,    'm/min'),
    -- Headbox
    ('PM1', 'headbox_pressure',       0.5,   3.5,   0.2,   'bar'),
    ('PM1', 'headbox_consistency',    0.2,   1.5,   0.1,   '%'),
    ('PM1', 'headbox_jet_speed',      400,   1250,  60,    'm/min'),
    -- Steam / Drying
    ('PM1', 'steam_pressure_p1',      2.0,   6.0,   0.5,   'bar'),
    ('PM1', 'steam_pressure_p2',      2.5,   7.0,   0.5,   'bar'),
    ('PM1', 'steam_pressure_p3',      3.0,   8.0,   0.5,   'bar'),
    ('PM1', 'dryer_temp_zone1',       80,    130,   5,     'degC'),
    ('PM1', 'dryer_temp_zone2',       90,    140,   5,     'degC'),
    ('PM1', 'dryer_temp_zone3',       100,   150,   5,     'degC'),
    -- Wire / Press
    ('PM1', 'wire_tension',           5,     30,    2,     'kN/m'),
    ('PM1', 'press_nip_load',         50,    350,   20,    'kN/m'),
    ('PM1', 'press_moisture_out',     40,    65,    3,     '%'),
    -- Stock Preparation
    ('PM1', 'pulp_consistency',       2.5,   5.0,   0.3,   '%'),
    ('PM1', 'refiner_power',          0,     4000,  200,   'kW'),
    ('PM1', 'broke_ratio',            0,     40,    5,     '%'),
    -- Coating (if applicable)
    ('PM1', 'coating_weight',         0,     25,    2,     'g/m2'),
    ('PM1', 'coating_speed',          400,   1100,  50,    'm/min')
ON CONFLICT (machine_id, variable_name) DO NOTHING;

-- ─── Machine Limits (Paper Machine PM2) ──────────────────────────────────────
INSERT INTO machine_limits (machine_id, variable_name, min_value, max_value, max_rate_of_change, unit) VALUES
    ('PM2', 'machine_speed',          350,   1100,  50,    'm/min'),
    ('PM2', 'headbox_pressure',       0.4,   3.2,   0.2,   'bar'),
    ('PM2', 'headbox_consistency',    0.2,   1.4,   0.1,   '%'),
    ('PM2', 'steam_pressure_p1',      2.0,   5.5,   0.5,   'bar'),
    ('PM2', 'steam_pressure_p2',      2.5,   6.5,   0.5,   'bar'),
    ('PM2', 'dryer_temp_zone1',       80,    125,   5,     'degC'),
    ('PM2', 'dryer_temp_zone2',       90,    135,   5,     'degC'),
    ('PM2', 'wire_tension',           5,     28,    2,     'kN/m'),
    ('PM2', 'press_nip_load',         50,    320,   20,    'kN/m'),
    ('PM2', 'pulp_consistency',       2.5,   4.8,   0.3,   '%'),
    ('PM2', 'refiner_power',          0,     3500,  200,   'kW'),
    ('PM2', 'broke_ratio',            0,     35,    5,     '%')
ON CONFLICT (machine_id, variable_name) DO NOTHING;

-- ─── Demo Transition Episodes (historical data for Qdrant seeding) ────────────
INSERT INTO transitions (episode_id, machine_id, from_recipe_id, to_recipe_id, started_at, ended_at, outcome, recovery_time_min, final_basis_weight, final_moisture, final_ash) VALUES
    ('e0000000-0000-0000-0000-000000000001', 'PM1',
        'b0000000-0000-0000-0000-000000000001',
        'b0000000-0000-0000-0000-000000000002',
        now() - interval '72 hours', now() - interval '71 hours',
        'success', 45.0, 80.2, 8.1, 10.1),
    ('e0000000-0000-0000-0000-000000000002', 'PM1',
        'b0000000-0000-0000-0000-000000000002',
        'b0000000-0000-0000-0000-000000000003',
        now() - interval '48 hours', now() - interval '46 hours 30 minutes',
        'offspec', 95.0, 91.5, 7.8, 8.5),
    ('e0000000-0000-0000-0000-000000000003', 'PM2',
        'b0000000-0000-0000-0000-000000000001',
        'b0000000-0000-0000-0000-000000000004',
        now() - interval '24 hours', now() - interval '23 hours',
        'success', 55.0, 100.1, 7.1, 6.2)
ON CONFLICT DO NOTHING;
