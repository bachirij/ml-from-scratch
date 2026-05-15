# Pipeline Orchestrator — Complete Guide

## What it is and why it exists

A production ML project is not a notebook. It has several distinct stages: fetching data,
cleaning it, building features, training a model, evaluating it, handling real-time updates,
detecting drift. Each of these stages lives in its own Python module under `src/`.

The problem: how do you trigger all of this cleanly from a terminal?

The answer: a `main.py` file at the root of the project that serves as a single entry point.
This is called a **pipeline orchestrator**.

Its role is simple:
- Know in what order to execute the steps
- Allow running only a subset of them
- Display what is happening and how long each step takes
- Centralize imports and configuration

`main.py` contains no business logic. It delegates everything to modules in `src/`.
It is a conductor, not a musician.

---

## What a CLI is

CLI stands for **Command Line Interface**.

Instead of opening an application with a window and buttons, you interact with the program
by typing commands in a terminal.

Concrete example:

```bash
# Without CLI — you modify a variable in the code, save, run
python main.py

# With CLI — you pass parameters directly in the command
python main.py --steps train --start-date 2023-01-01 --end-date 2023-12-31
```

The elements of a CLI command:

```
python main.py --steps train evaluate --start-date 2023-01-01
│              │        │             │
│              │        │             └── value of the --start-date argument
│              │        └── values of the --steps argument (multiple allowed)
│              └── named argument (starts with --)
└── Python script to execute
```

**Arguments** start with `--` (double dash). They have a name and a value.
Some accept multiple values (like `--steps train evaluate`).
Some have default values and are optional.

In Python, the standard module for building a CLI is called `argparse`.

---

## File structure

```
main.py
│
├── Header docstring             → documentation, usage examples
├── Configuration                → CONFIG dict, ALL_STEPS, default dates
├── import_modules()             → centralized imports
├── Step functions               → one function per pipeline stage
├── STEP_FUNCTIONS               → registry: step name → function
├── run_pipeline()               → main orchestrator
├── parse_args()                 → CLI definition with argparse
└── __main__ block               → entry point, validation, call
```

Each section has a precise role. Let's walk through them in order.

---

## Section 1 — The header docstring

```python
"""
<project-name> - Pipeline Orchestrator
=======================================

Single entry point to run all or part of the ML pipeline.

Usage:
------
    python main.py
    python main.py --steps ingest
    ...
"""
```

The first thing in the file is a long docstring that explains:
- What the file does
- How to use it with concrete examples
- Special behaviors of each step

This is the documentation you will read six months later when you have forgotten how it works.
It also appears when you run `python main.py --help`.

---

## Section 2 — Configuration

```python
CONFIG = {
    "entity": "default",
    "raw_data_path": "data/raw/",
    "processed_data_path": "data/processed/",
    "features_path": "data/features/",
    "models_path": "models/",
    "reports_path": "reports/",
    "monitoring_path": "data/monitoring/",
}

ALL_STEPS = [
    "ingest", "preprocess", "features",
    "train", "evaluate", "realtime", "monitor", "retrain",
]

DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE   = str(datetime.now().year - 1) + "-12-31"
```

`CONFIG` is a dictionary that centralizes everything that can vary from one project to another:
paths to data folders, the name of the entity being processed (a country, a client, a sensor),
global parameters. Rather than repeating `"data/raw/"` in five places across the codebase,
you define it once here.

`ALL_STEPS` is the ordered list of all possible pipeline stages. The order matters: it is the
order in which they will be executed when you run `python main.py` without arguments.

`DEFAULT_END_DATE` uses `datetime.now().year - 1` to always point to the most recent complete
year, without having to edit the file every year.

---

## Section 3 — import_modules()

```python
def import_modules() -> dict:
    try:
        from src.ingestion.ingest import fetch_and_store
        from src.modeling.train import run_training
        # ...

        return {
            "fetch_and_store": fetch_and_store,
            "run_training":    run_training,
            # ...
        }

    except ImportError as e:
        print(f"\n[ERROR] Failed to import a module: {e}")
        print("  → Make sure you run main.py from the project root.")
        sys.exit(1)
```

All imports are grouped into a single function, for two reasons.

**First reason: error handling.** If a module is missing or a dependency is not installed,
Python raises an `ImportError`. By catching this exception here, you display a clear message
(`Failed to import a module: ...`) instead of a cryptic traceback. This is the difference
between `ModuleNotFoundError: No module named 'lightgbm'` and a message that tells you
exactly what to do.

**Second reason: readability.** Imports do not clutter the top of the file with dozens of lines.
They appear where they are actually needed.

The function returns a dictionary `{"name": function}`. Each step receives this dictionary
and calls the function it needs via `modules["function_name"](...)`.

---

## Section 4 — Step functions

Each pipeline stage has its own function. They all share the same signature:

```python
def step_ingest(config: dict, start_date: str, end_date: str, modules: dict):
    ...

def step_train(config: dict, start_date: str, end_date: str, modules: dict):
    ...
```

**Uniform signature**: the same four parameters in the same order for every function.
This allows the orchestrator to call them identically regardless of the step.

Each function does exactly one thing:
1. Print what it is doing (`print("\n--- ... ---")`)
2. Call the corresponding module function with the right arguments

```python
def step_ingest(config: dict, start_date: str, end_date: str, modules: dict):
    print("\n--- Raw data ingestion ---")
    modules["fetch_and_store"](
        entity=config["entity"],
        start_date=start_date,
        end_date=end_date,
        output_path=config["raw_data_path"],
    )
```

It contains no business logic. All logic lives in `src/ingestion/ingest.py`.
`step_ingest` is just the wiring between the CLI and the module.

The docstring of each step explains:
- What it does concretely
- Which arguments it uses and which it ignores
- What to adapt for a new project

### The eight standard steps

| Step | What it does | Uses date args? |
|---|---|---|
| `ingest` | Fetch raw data from external sources (APIs, databases, files) | Yes |
| `preprocess` | Clean, merge, validate raw sources into a single dataset | Yes |
| `features` | Build model-ready features (lags, encodings, rolling stats) | Yes |
| `train` | Train models, select the best, save artifacts | No — managed by config |
| `evaluate` | Run saved model on held-out test set, generate report | No — managed by config |
| `realtime` | Fetch recent data window, run model, store predictions | No — fixed window |
| `monitor` | Compare current distributions against training reference | No |
| `retrain` | Full retrain from scratch when drift is detected | No — managed by config |

Steps that ignore date arguments have their own time window logic defined in their respective
modules (`src/modeling/config.py`, `src/monitoring/monitor.py`, etc.).

---

## Section 5 — The STEP_FUNCTIONS registry

```python
STEP_FUNCTIONS = {
    "ingest":     step_ingest,
    "preprocess": step_preprocess,
    "features":   step_features,
    "train":      step_train,
    "evaluate":   step_evaluate,
    "realtime":   step_realtime,
    "monitor":    step_monitor,
    "retrain":    step_retrain,
}
```

This dictionary maps a step name (string) to the corresponding Python function.

The benefit: the orchestrator does not need to know the steps individually. It receives a list
of names (`["train", "evaluate"]`) and simply does:

```python
for step_name in steps:
    STEP_FUNCTIONS[step_name](...)
```

To add a new step: create the function, add it to this dictionary. Two places to edit, that is
all. The orchestrator does not change.

---

## Section 6 — run_pipeline()

```python
def run_pipeline(steps: list, config: dict, start_date: str, end_date: str):
    modules = import_modules()

    # Print header
    print("=" * 60)
    print(f"  Steps : {' → '.join(steps)}")
    # ...

    pipeline_start = time.time()

    for i, step_name in enumerate(steps, start=1):
        print(f"  STEP {i}/{len(steps)} : {step_name.upper()}")

        step_start = time.time()
        STEP_FUNCTIONS[step_name](
            config=config,
            start_date=start_date,
            end_date=end_date,
            modules=modules,
        )
        elapsed = time.time() - step_start
        print(f"[OK] '{step_name}' completed in {elapsed:.1f}s")

    # Print footer with total time
```

This is the core of the file. It:
1. Imports all modules once at the beginning
2. Prints a header with execution parameters
3. Loops over the requested steps in order
4. Times each step individually
5. Prints a footer with total elapsed time

`time.time()` before and after each step gives you elapsed time in seconds.
`enumerate(steps, start=1)` gives you a counter starting at 1 for display (`STEP 1/3`, etc.).

---

## Section 7 — parse_args()

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ML Pipeline Orchestrator",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="..."
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=ALL_STEPS + ["all"],
        metavar="STEP",
        help="..."
    )

    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        dest="start_date",
        help="..."
    )

    return parser.parse_args()
```

`argparse.ArgumentParser` creates the CLI. Each argument is declared with `add_argument`.

The key parameters of `add_argument`:

| Parameter | Role | Example |
|---|---|---|
| `"--steps"` | Argument name (with `--` prefix) | `--steps train` |
| `nargs="+"` | Accepts one or more values | `--steps train evaluate` |
| `default` | Value used if the argument is not provided | `default=["all"]` |
| `choices` | List of allowed values (argparse validates automatically) | `choices=ALL_STEPS` |
| `dest` | Variable name in the returned namespace | `dest="start_date"` |
| `help` | Text displayed by `--help` | `help="Steps to run"` |

`parser.parse_args()` reads `sys.argv` (everything typed after `python main.py`), matches it
against the declared arguments, validates choices, applies defaults, and returns an
`argparse.Namespace` object where each argument is an attribute:

```python
args = parse_args()
args.steps       # ["train", "evaluate"]
args.start_date  # "2023-01-01"
args.end_date    # "2023-12-31"
```

`formatter_class=argparse.RawTextHelpFormatter` preserves line breaks in help text,
so your `epilog` with usage examples displays correctly.

---

## Section 8 — The __main__ block

```python
if __name__ == "__main__":
    args = parse_args()

    steps = ALL_STEPS if "all" in args.steps else args.steps

    # Validate date format
    for date_str, label in [(args.start_date, "--start-date"), (args.end_date, "--end-date")]:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print(f"[ERROR] {label} must be ISO format YYYY-MM-DD, got: '{date_str}'")
            sys.exit(1)

    if args.start_date > args.end_date:
        print(f"[ERROR] --start-date must be <= --end-date")
        sys.exit(1)

    run_pipeline(
        steps=steps,
        config=CONFIG,
        start_date=args.start_date,
        end_date=args.end_date,
    )
```

`if __name__ == "__main__"` is a Python convention. It ensures that this block only runs when
the file is executed directly (`python main.py`), not when it is imported by another module.

This block:
1. Parses CLI arguments
2. Resolves `"all"` into the full step list
3. Validates date format and logical order before doing anything
4. Calls `run_pipeline()` with all resolved parameters

Input validation happens here, before any module is imported or any step runs. Fail fast with
a clear message rather than letting an invalid date propagate deep into the pipeline.

---

## How to adapt this template to a new project

**Step 1 — Update CONFIG**

Replace paths and entity name to match your project structure:

```python
CONFIG = {
    "entity": "my_project",
    "raw_data_path": "data/raw/",
    ...
}
```

**Step 2 — Update ALL_STEPS**

Remove steps you do not need, add project-specific ones:

```python
ALL_STEPS = ["ingest", "preprocess", "features", "train", "evaluate"]
# Removed: realtime, monitor, retrain — not needed for a batch-only project
```

**Step 3 — Update import_modules()**

Replace the imports with your actual module paths:

```python
from src.ingestion.my_fetcher import fetch_data
from src.modeling.my_trainer import train_model
```

**Step 4 — Update step functions**

Adapt each step function to pass the right arguments to your modules.
Each step has a docstring section `Adapt:` that lists what to change.

**Step 5 — Update STEP_FUNCTIONS**

Add or remove entries to match your updated step list.

**Step 6 — Add CLI arguments if needed**

If your project has additional parameters (model type, environment, entity id),
add them in `parse_args()` and thread them through `CONFIG` or as direct arguments.

---

## Common patterns

### Running a single step in isolation

```bash
python main.py --steps train
```

Useful when data is already preprocessed and you only want to retrain.

### Chaining steps without going back to the start

```bash
python main.py --steps features train evaluate
```

Skips ingestion and preprocessing — assumes clean data already exists.

### Processing a specific time window

```bash
python main.py --steps ingest preprocess --start-date 2024-01-01 --end-date 2024-12-31
```

Re-ingests and reprocesses only 2024 data without touching the rest.

### Forcing a manual drift check

```bash
python main.py --steps monitor
```

### Triggering a full retrain after detected drift

```bash
python main.py --steps retrain evaluate
```

---

## Project structure this file assumes

```
project-root/
│
├── main.py                          ← this file
│
├── src/
│   ├── ingestion/
│   │   ├── ingest.py                ← fetch_and_store()
│   │   └── realtime.py              ← fetch_and_store_realtime()
│   ├── preprocessing/
│   │   └── preprocess.py            ← build_processed_dataset()
│   ├── feature_engineering/
│   │   └── build_features.py        ← build_features()
│   ├── modeling/
│   │   ├── config.py                ← temporal splits, model params
│   │   ├── train.py                 ← run_training()
│   │   ├── evaluate.py              ← run_evaluation()
│   │   └── retrain.py               ← run_retraining()
│   └── monitoring/
│       └── monitor.py               ← run_monitoring()
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── monitoring/
│
├── models/
├── reports/
└── scheduler.py                     ← automated real-time execution (cron)
```

Each module in `src/` exposes a single public function that the corresponding step in
`main.py` calls. Internal logic, helper functions, and classes stay inside their module.