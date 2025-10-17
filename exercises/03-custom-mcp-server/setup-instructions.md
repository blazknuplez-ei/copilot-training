
# Exercise 03 (Setup): Install Synth, generate data, and load into SQLite

This guide gets you ready for the MCP server build by helping you:
- Install Synth (the data generator)
- Generate sample airline data
- Load the data into the SQLite database

Who is this for?
- macOS/Linux users installing Synth directly
- Windows users using Docker (or WSL) for Synth

At a glance (4 steps):
1) Install/verify Synth
2) Generate data to `data/synthetic_output/generated_data.json`
3) Load into SQLite with `load_all()`
4) Verify row counts (passengers, routes, discounts)

Before you start:
- Use the repository root as your working directory
- Ensure Python and pip are available (recommended: use the provided virtual environment)
- If you’re on Windows, you can jump to Part 5 (Docker method) or use WSL

## Part 1: Install and Test Synth CLI

### Step 1: Install Synth

What is Synth?

Synth is a command-line tool that generates realistic synthetic data from JSON schema templates. Instead of hand-writing records, you define a schema (a "recipe") and Synth produces many realistic records using faker-like providers.

Install Synth on macOS / Linux:

```bash
curl -sSL https://getsynth.com/install | sh
```

Verify the installation:

```bash
synth version
```

You should see output like `synth 0.6.9` (version may vary).

If `synth` is not found after install, make sure your PATH includes the install directory and restart your shell.

### Step 2: Generate a small sample dataset

From the repository root run a small generation to confirm everything works:

```bash
cd airline-discount-ml

# Generate 10 records per collection and write to the output JSON
synth generate synth_models/airline_data --size 10 --seed 42 > data/synthetic_output/generated_data.json
```

What this command does:
- `synth generate` runs the generator
- `synth_models/airline_data` points to the folder with the provided schemas
- `--size 10` creates 10 records per collection (passengers, routes, discounts)
- `--seed 42` fixes randomness so runs are reproducible
- `> data/synthetic_output/generated_data.json` saves the combined JSON output

Confirm the file exists and inspect the structure:

```bash
ls -lh data/synthetic_output/

# Pretty-print passengers section (example)
cat data/synthetic_output/generated_data.json | python -m json.tool | grep -A 10 '"passengers"'

# Pretty-print routes section
cat data/synthetic_output/generated_data.json | python -m json.tool | grep -A 8 '"routes"'

# Pretty-print discounts section
cat data/synthetic_output/generated_data.json | python -m json.tool | grep -A 8 '"discounts"'
```

Example passenger entry you'll see:

```json
	"passengers": [
		{
			"id": 76496,
			"name": "Germaine Collins",
			"travel_history": {
				"total_spend": 24718.08,
				"trips": 4
			}
		},
```

## Part 2: Load Data into SQLite Database

Your models expect data in SQLite. Use the provided loader script to convert the generated JSON into the database schema.

Run the loader (from repository root):

```bash
python -c "from src.data.load_synthetic_data import load_all; load_all()"
```

Expected loader output (example):

```
✓ Database connection successful: /Users/maria/airlst-github-copilot-training/airline-discount-ml/data/airline_discount.db
✓ Cleared existing data from all tables
✓ Loaded 10 passengers
✓ Loaded 5 routes
✓ Loaded 100 discounts

✅ Database populated with synthetic data!
   Total: 10 passengers, 5 routes, 100 discounts
```

If you see a foreign key error during loading, it usually means discounts referenced IDs not present in routes or passengers; regenerating with the same schema generally fixes this (see regenerate section below).

### Quick verification

```bash
python -c "from src.data.database import get_connection; import pandas as pd; db = get_connection(); print('Passengers:', pd.read_sql('SELECT COUNT(*) as c FROM passengers', db.connection)['c'][0]); print('Routes:', pd.read_sql('SELECT COUNT(*) as c FROM routes', db.connection)['c'][0]); print('Discounts:', pd.read_sql('SELECT COUNT(*) as c FROM discounts', db.connection)['c'][0]); db.close()"
```

Expected output:

```
Passengers: 10
Routes: 5
Discounts: 100
```

## Part 3: Optional — Explore the Data

Open the exploratory notebook to visualize distributions and samples:

```bash
jupyter lab notebooks/exploratory_analysis.ipynb
```

Or run small SQL queries from the command line to inspect records:

```bash
python -c "from src.data.database import get_connection; import pandas as pd; db = get_connection(); print('=== Sample Passengers ==='); print(pd.read_sql('SELECT name, travel_history FROM passengers LIMIT 5', db.connection)); print('\n=== Top Discounted Routes ==='); q='''SELECT r.origin, r.destination, d.discount_value FROM discounts d JOIN routes r ON d.route_id = r.id ORDER BY d.discount_value DESC LIMIT 5'''; print(pd.read_sql(q, db.connection)); db.close()"
```

## Part 4: Regenerating Data

To create larger datasets or different random draws:

```bash
# Generate 1000 records per collection with a different seed
synth generate synth_models/airline_data --size 1000 --seed 99 > data/synthetic_output/generated_data.json

# Reload into SQLite
python -c "from src.data.load_synthetic_data import load_all; load_all()"
```

Use cases:
- Stress testing models
- Generating edge-case scenarios (lots of long-distance passengers, many high-spend travelers, etc.)

## Part 5: Windows Installation (Docker Method)

If you're on Windows, Synth may not have a native installer. Use the official Docker image instead.

1. Install Docker Desktop for Windows: https://www.docker.com/products/docker-desktop

2. Pull the Synth image:

```powershell
docker pull getsynth/synth
```

3. Run Synth to generate JSON (PowerShell example):

```powershell
cd airline-discount-ml

docker run --rm `
  -v ${PWD}/synth_models:/synth_models `
  -v ${PWD}/data:/data `
  getsynth/synth generate /synth_models/airline_data `
  --to json:///data/synthetic_output/ `
  --size 1000 `
  --seed 42
```

4. Load the generated data into SQLite (same as macOS/Linux):

```powershell
python -c "from src.data.load_synthetic_data import load_all; load_all()"
```

### Alternative: Use WSL

If you have WSL installed, you can install Synth inside WSL and follow the Linux instructions:

```bash
# Inside WSL (Ubuntu) shell
curl -sSL https://getsynth.com/install | sh
synth --version
```

Then follow the macOS/Linux steps above.

---

When these setup steps are complete, continue with the main exercise file which focuses on building the MCP server and adding Synth tools.

Next: Build the MCP server
- Open: `exercises/03-custom-mcp-server/exercise-03-custom-mcp-server-instructions.md`
- You’ll add endpoints, run the server locally, and connect it to VS Code

