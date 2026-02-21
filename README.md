# SSE_P1
## Quick Start with `sudo -s`

```bash
# 1. Install
chmod +x scripts/setup.sh
./scripts/setup.sh

# 2. Generate test files
source .venv/bin/activate
python3 scripts/generate_test_data.py

# 3. Run full experiment (~4-6 hours with 30 reps)
python3 scripts/run_experiment.py

# 4. Analyze (produces plots + stat tests) - Pending
python3 analysis/analyze.py

# 5. Generate blog tables - Pending
python3 analysis/generate_tables.py