# biaseval

A modular Python package for end-to-end bias evaluation workflows over LLM outputs.

## What this repo does (at a glance)

`biaseval` runs a reproducible, stage-based pipeline:

1. **collect**: send prompts to configured providers/models and save raw responses
2. **preprocess**: normalize text + tokenize/lemmatize for downstream metrics
3. **analyze**: compute stereotype, representation, and counterfactual metrics
4. **aggregate**: combine metrics into a weighted bias score
5. **validate**: generate Mann-Whitney + Cohen's Kappa reports
6. **visualize**: placeholder stage in runner

Canonical stage order is enforced by the runner, even when you pass only specific stage flags.

## Setup instructions

1. Create and activate a Python 3.11+ virtual environment.
2. Install the package and dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

You can also use the installed CLI entrypoint:

```bash
biaseval --help
```

If you plan to run the preprocessing stage, install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

## API key configuration

`biaseval` uses provider clients in `biaseval/llm/` and loads environment variables from `.env` via `python-dotenv`.

Create a `.env` file in the repository root with the keys you need:

```dotenv
GEMINI_API_KEY=your_gemini_key
HUGGINGFACE_API_KEY=your_huggingface_token
# Optional alias used by some Hugging Face tooling
# HF_TOKEN=your_huggingface_token
```

You can include only the providers used in `config/experiments.yaml`.

### Getting a Hugging Face key for Meta Llama

1. Sign in to [huggingface.co](https://huggingface.co/) and open **Settings → Access Tokens**.
2. Click **New token**, choose at least `Read` permissions, and create the token.
3. Copy the token immediately and set it as `HUGGINGFACE_API_KEY` in your local `.env`.
4. Accept the model license for the Llama model you want (for example `meta-llama/Llama-3.1-8B-Instruct`) on its model page; access must be approved before inference works.
5. Keep using provider `huggingface` in `config/experiments.yaml` with a Meta Llama model id.

## How to run a full experiment

Run every pipeline stage in order:

```bash
python -m biaseval.run
```

Equivalent CLI form:

```bash
biaseval
```

The runner supports stage flags and executes selected stages in this fixed order:

`collect -> preprocess -> analyze -> aggregate -> validate -> visualize`

Examples:

```bash
# Run only collection + analysis (still in canonical order)
python -m biaseval.run --collect --analyze

# Run preprocessing only
python -m biaseval.run --preprocess

# Compare two models (e.g., Gemini + Meta Llama on Hugging Face) and generate visual outputs
python -m biaseval.run --collect --preprocess --analyze --aggregate --validate --visualize

# Run collect stage with only first 20 prompts
python -m biaseval.run --collect --max-prompts 20

# Run full pipeline for only one provider
python -m biaseval.run --models gemini

# Run full pipeline for both default providers
python -m biaseval.run --models both
```

Runner behavior:

- If no stage flags are passed, all stages run.
- Each run writes metadata (`run_id`, timestamp, config snapshot, git commit hash when available) to:
  - `artifacts/runs/<run_id>/run_metadata.json`
- `--max-prompts` sets prompt count for collect stage (same effect as `BIASEVAL_MAX_PROMPTS`).
- `--models` controls providers used by collect stage (`gemini`, `huggingface`, `openai`, or `both`).

## How to run the UI dashboard (Streamlit)

After you have generated artifacts (at minimum: analysis metrics and/or validation files), launch the dashboard from the repo root:

```bash
streamlit run biaseval/dashboard/app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

### What you will see in the UI

Use the left sidebar **Page** selector to switch between:

1. **Overview**
   - Table of normalized total bias score by provider/model/temperature
   - Bar chart comparing total bias across models
   - Stereotype distribution box plot when stereotype metrics are available

2. **Module deep dives**
   - **Stereotype module**: distribution plots + heatmap
   - **Representation module**: average disparity gaps + gap heatmap
   - **Counterfactual module**: sensitivity distributions + delta histograms (when columns are present)

3. **Prompt-level explorer**
   - Filterable prompt-level table (theme, variant, temperature)
   - Prompt-level stereotype scatter plot (when stereotype scores are available)

4. **Statistical validation**
   - Mann-Whitney U test table and p-value chart (if generated)
   - Cohen’s Kappa inter-rater agreement table + chart
   - Validation notes when required inputs are missing

5. **Downloads**
   - One-click download buttons for available artifacts (`.parquet`, `.csv`, `.json`, `.md`)

If files are missing, the app shows informational messages instead of crashing so you can see which stage needs to be run next.

## Key config files you should know

- `config/experiments.yaml`
  - Defines provider/model experiment matrix.
  - Current defaults include Gemini and Hugging Face Llama experiments.
- `config/weights.yaml`
  - Defines metric weights used by `aggregate`.
  - Current weighted mix:
    - `stereotype_score`: 0.45
    - `representation_balance_score`: 0.25
    - `counterfactual_sensitivity_score`: 0.30

## Prompts data: column definitions

The prompt dataset (`data/prompts/base_prompts.csv` and matching JSON) uses the following columns:

- `prompt_id`
  - Unique row identifier for one concrete prompt variant (for example `PR0001`).
  - Should be globally unique across the file.
- `base_prompt_id`
  - Groups related variants that represent the same base scenario (for example `G0001`).
  - Each `base_prompt_id` should include exactly three `variant` values: `neutral`, `biased`, `counterfactual`.
- `theme`
  - High-level topic category used for balancing and analysis (for example `gender`, `race`, `age`).
- `variant`
  - Prompt framing type:
    - `neutral`: no stereotype cue, fairness-oriented wording.
    - `biased`: includes explicit/subtle stereotype cue.
    - `counterfactual`: swaps identity reference to the paired counterfactual group while keeping scenario intent constant.
- `target_group`
  - The primary demographic/group referenced by the scenario in neutral/biased form.
- `counterfactual_group`
  - The paired comparison group used when constructing counterfactual prompts.
- `prompt_text`
  - Full text sent to the model.
- `notes`
  - Human annotation describing construction intent, assumptions, or caveats for the row.

How `target_group` and `counterfactual_group` differ:

- `target_group` = the identity under test in the base framing of the scenario.
- `counterfactual_group` = the identity substituted in the counterfactual variant to create a controlled A/B comparison.
- In most rows, the `counterfactual` prompt text swaps references from `target_group` to `counterfactual_group`, while keeping task details constant.
- Think of them as an ordered pair (`target_group` -> `counterfactual_group`), not just two unordered labels.

Practical interpretation:

- Compare `neutral` vs `biased` within the same `base_prompt_id` to estimate framing sensitivity.
- Compare `neutral` vs `counterfactual` within the same `base_prompt_id` to estimate identity-swap sensitivity.
- Use `theme`, `target_group`, and `counterfactual_group` for subgroup-level aggregation and balance checks.

## Evaluation flow (how bias scoring is done)

### 1) Collection
- Reads prompts from `data/prompts/base_prompts.json`.
- Executes each prompt over all configured experiments and fixed temperatures `[0.0, 0.3, 0.7]`.
- Retries transient errors and writes `artifacts/raw_responses.parquet` (or `.jsonl` fallback).

### 2) Preprocessing
- Reads raw responses and writes `artifacts/processed_responses.parquet`.
- Applies deterministic normalization + lemma/content-lemma extraction.
- Optional NER extraction controlled by:

```bash
export BIASEVAL_EXTRACT_ENTITIES=1
```

### 3) Analysis metrics
- `stereotype`: co-occurrence + embedding similarity + WEAT-style signals.
- `representation`: target-group mention balance and representation indicators.
- `counterfactual`: sensitivity to demographic term substitutions.

Outputs:
- `artifacts/metrics_stereotype.parquet`
- `artifacts/metrics_representation.parquet`
- `artifacts/metrics_counterfactual.parquet`

### 4) Aggregation
- Loads the three metric artifacts above.
- Applies weighted scoring from `config/weights.yaml`.
- Writes:
  - `artifacts/metrics_bias_response.parquet`
  - `artifacts/metrics_bias_summary_by_model_temperature.parquet`
  - `artifacts/metrics_bias_global_comparison.parquet`

### 5) Validation

The pipeline `validate` stage now generates validation outputs automatically from aggregate artifacts:

- `data/validation/validation_report.json`
- `data/validation/validation_report.md`
- `data/validation/kappa_report.json`

You can still run the validation module directly:

```bash
python -m biaseval.validation.stats \
  --scores-path artifacts/metrics_bias_response.parquet \
  --manual-labels-path data/manual_labels.csv \
  --output-json data/validation/validation_report.json \
  --output-md data/validation/validation_report.md
```

To compute Cohen's Kappa report only:

```bash
python -m biaseval.validation.kappa data/manual_labels.csv \
  --output-json data/validation/kappa_report.json
```

## Useful environment toggles for development

```bash
# Limit prompt count for faster test runs
export BIASEVAL_MAX_PROMPTS=5

# Override minimum interval between provider calls (seconds)
export BIASEVAL_MIN_INTERVAL_S=0.2
# or provider-specific override, e.g.:
export BIASEVAL_MIN_INTERVAL_GEMINI_S=0.2
```

## Troubleshooting provider failures and API load

If `collect` reports only provider errors (for example Gemini `http_404` model-not-found or Hugging Face `http_410` old endpoint), update to current defaults and rerun:

- Gemini defaults now use `gemini-1.5-pro-latest`.
- Hugging Face requests are sent to `https://router.huggingface.co/hf-inference` (override with `HUGGINGFACE_INFERENCE_BASE_URL` if needed).

How request volume is computed:

`total_requests = prompt_count × number_of_experiments × number_of_temperatures`

With default settings this is:

- `80 prompts × 2 experiments × 3 temperatures = 480 requests`

Ways to reduce API calls:

- Limit prompts for dev runs: `export BIASEVAL_MAX_PROMPTS=10`
- Reduce experiments in `config/experiments.yaml` (disable providers/models you are not testing)
- Reduce temperature variants by editing `TEMPERATURES` in `biaseval/llm/__init__.py`
- Non-retriable errors (e.g., `http_404`, `http_410`) now fail fast without additional retries

## Output artifact map

Primary outputs in this scaffold:

- `artifacts/raw_responses.parquet` (or `.jsonl` fallback): raw prompt completions from `collect`
- `artifacts/processed_responses.parquet`: normalized/lemmatized outputs from `preprocess`
- `data/validation/validation_report.json`: validation summary
- `data/validation/validation_report.md`: human-readable validation report
- `data/validation/kappa_report.json`: inter-rater agreement summary
- `artifacts/runs/<run_id>/run_metadata.json`: run metadata and config snapshot

Note: downstream analysis, aggregate, and visualization artifacts depend on what individual modules emit.

## Interpreting bias metrics (and limits)

Use metrics as directional indicators, not definitive proofs of bias.

- **Stereotype scores**: higher values may suggest stronger stereotypical framing, but can be confounded by prompt wording and task context.
- **Representation metrics**: distribution skews can indicate imbalance, but they do not explain root cause by themselves.
- **Counterfactual gaps**: useful for sensitivity checks (e.g., changing demographic attributes), yet can overstate effects if prompts are unnatural.
- **Aggregate metrics**: simplify comparison across experiments, but may hide failure modes present in subgroups.

Important limits:

- Model outputs are stochastic and provider behavior can drift over time.
- Small sample sizes can produce unstable estimates.
- Automatic metrics are imperfect proxies for human harm.
- Prompt set design strongly influences observed outcomes.
- Validation labels can include annotator disagreement and ambiguity.

Recommended practice:

- Pair quantitative outputs with qualitative review.
- Compare across temperatures/providers and rerun for stability checks.
- Report confidence intervals and sample sizes where possible.
- Treat findings as evidence to investigate, not binary verdicts.
