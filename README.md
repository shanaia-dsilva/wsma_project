# Between the Lines

A complete local demo for **Evaluating Popularity-Aware Personalized Reranking for Book Discovery on Goodreads Data** (WSMA CSE 3132).

## Run the demo

Requires Python 3.10+ and the two packages in requirements.txt. No API key, cloud service, or Node build is needed.

```sh
cd /Users/shanaiadsilva/Documents/ChatGPT/WSMA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python server.py
```

Open **http://localhost:8000**. Stop with Ctrl+C. If Python already has NumPy and SciPy installed, simply run `python3 server.py`. Use `--port 8001` if needed. The server binds only to the local machine.

## What works

- Responsive discovery UI with original typographic book covers and recommendation explanations.
- 60 curated book records across five genres, plus 360 deterministic synthetic readers.
- Popularity baseline, shrinkage-regularized implicit item-item collaborative filtering, fixed novelty reranking, and history-adapted reranking.
- Reader switching, genre filtering, discovery-strength slider, and optional personalization.
- Searchable catalogue, book details, saved bookshelf, and custom reading history. Browser-local storage persists your shelf and reading history.
- Live Model Lab: Precision@10, Recall@10, NDCG@10, catalogue coverage, long-tail share, average training popularity, Gini concentration, validation sensitivity, reader-group analysis, and CSV export.
- Real-data importer with provenance hashes and documented filtering.
- Reproducible experiment exports and automated correctness tests.

## Data honesty

**Bundled reading interactions are synthetic, not Goodreads data.** They are generated with NumPy seed 42 from genre affinity and a popularity prior. Titles and authors are real; summaries and covers are demo content. No ratings, real user activity, or research findings are fabricated. The UI labels this explicitly. Synthetic metrics demonstrate the pipeline and must not be presented as Goodreads study results.

The official dataset is at https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html. Its provider specifies academic use and prohibits redistribution. Obtain an authorized copy separately; no dataset download or redistribution is performed by this project.

## Present the demo in five minutes

1. Open Discover. Choose Reader 003 to show how a fantasy-oriented reader receives a different list.
2. Compare Most popular with Collaborative filtering. Books in the reader's training history are excluded.
3. Select Discovery reranking and move discovery strength from 0% to 70%. Show how the actual ranking changes.
4. Enable history adaptation; explain that the effective strength depends on the reader's training-history tail share.
5. Search the catalogue for Piranesi, open its details, save it, and click “I've read this.” Repeat with two other favourites.
6. Open My bookshelf and choose Discover with my taste. These recommendations use your selected books as seeds; saved books alone do not change your taste.
7. Open Model Lab. Explain relevance alongside coverage and long-tail share. Export the comparison CSV.
8. State clearly that this demonstration uses synthetic interactions. Use imported real data for the final empirical project.

## Model and experimental protocol

- Interactions are binary implicit positives. Ratings are not used.
- Deduplication precedes splitting. Each user receives a deterministic random 15% validation and 15% test holdout, rounded down with a minimum of one each. The remainder trains the model. This is not a temporal split.
- Training counts alone define popularity, similarity, novelty, user groups, and head/tail classification.
- Similarity: cosine of binary item co-interaction vectors, multiplied by `co_count / (co_count + 10)`. Self-similarity is zero. SciPy sparse matrices avoid constructing the full dense user-item matrix.
- Scores sum similarities to the user's training history. The top 40 eligible unseen candidates are retained.
- Relevance is min-max normalized within the candidate list; an equal-score list receives zero relevance.
- `novelty(i) = 1 - log(1 + training_count(i)) / max_j log(1 + training_count(j))`.
- `score(u,i) = (1 - lambda_u) * relevance(u,i) + lambda_u * novelty(i)`.
- Optional personalization: `lambda_u = clip(lambda * (0.5 + user_training_tail_fraction), 0, 0.8)`.
- Head = top 20% of training-observed books, rounded up; tail = remainder. Count ties use catalogue index. Unobserved books are not recommendation candidates.
- Validation sweeps 0.0 through 0.8 by 0.1. Select the highest tail share retaining at least 95% of CF validation NDCG. Ties use NDCG, then lower lambda. This rule is fixed before test evaluation.
- The personalized variant uses the fixed variant's selected strength and is not separately tuned.
- Final test uses the unchanged training model, excludes both train and validation items, and evaluates the test holdout. Validation positives are not added to training. Test metrics are cached per server run.
- Precision always divides hits by 10, even if fewer than 10 eligible recommendations exist. Recall and ideal DCG include all held-out positives; cold/unseen test items are therefore misses.
- Coverage and Gini include all training-observed items, including zero-exposure items.
- Niche readers have at least 60% tail items in their training history; other readers are the mainstream group. These are descriptive groups, not sensitive traits.
- The interactive slider defaults to 0.30 for exploration; Model Lab reports the independently validation-selected setting. Interactive custom histories do not modify offline results.

## Import real data

Prefer a genre-specific pair of **metadata JSONL(.gz)** and **interaction JSONL(.gz)** files. The complete UCSD interaction CSV uses mapped IDs: convert those through `book_id_map.csv` before using the generic CSV importer. Supplying unmatched ID spaces will not produce meaningful joins.

```sh
python scripts/import_data.py \
  --books /path/to/goodreads_books_poetry.json.gz \
  --interactions /path/to/goodreads_interactions_poetry.json.gz \
  --genre Poetry \
  --max-books 1000 --max-users 1000 --max-records 1000000 \
  --min-user-items 8 --output data/imported/poetry
python server.py --data data/imported/poetry
```

Canonical CSV is also accepted:

- books: `book_id,title,author,genre,description,publication_year`
- interactions: `user_id,book_id` (additional fields ignored)

The importer retains the first valid distinct metadata records, admits the first users encountered within the scan limit, deduplicates pairs, filters users below the minimum, then drops unused metadata. This is a bounded convenience sample, **not representative random sampling**. All valid shelf interactions count as positive, irrespective of `is_read` or rating. Timestamps are not parsed. Review this policy before conducting the real study.

Raw user IDs are discarded from output. Goodreads metadata often contains author IDs rather than names; the importer labels those honestly until you enrich them from the authorized author metadata. The manifest records filenames, SHA-256 hashes, limits, counts, and filtering policy. Output must be an empty/new directory to avoid overwriting a dataset. At least 10 eligible readers are required. Dense subsets are preferable; increase limits if too few users survive.

Restart the server after importing. Large inputs are bounded to 5,000 books / 10,000 admitted readers; similarity computation and exhaustive offline evaluation can still be expensive. This is an undergraduate-scale local prototype, not a distributed recommender.

## Reproduce results and tests

```sh
python -m unittest discover -s tests -v
python scripts/evaluate.py --output outputs/demo
# Real subset:
python scripts/evaluate.py --data data/imported/poetry --output outputs/poetry
```

Exports: `results.json` (metrics and protocol), `comparison.csv`, `validation-sweep.csv`, `reader-groups.csv`, `splits.json`, and `sensitivity.svg`. `experiment.ipynb` provides a notebook entry point. Data hashes and the seed allow provenance checks. Pin your actual Python/package versions for a final submission; this run used Python 3.12, NumPy 1.26.4, SciPy 1.13.1.

## Project layout

```text
server.py                 Local HTTP/API server
model.py                  Data generation, splitting, CF, reranking, evaluation
data/catalog.json         Curated demo book metadata
static/                   HTML, CSS, JavaScript frontend
scripts/import_data.py    Bounded JSONL/gzip/CSV importer
scripts/evaluate.py       Reproducible CSV/JSON/SVG exports
experiment.ipynb          Notebook entry point
tests/test_model.py       Model and metric correctness checks
tests/test_import.py      Import pipeline integration checks
```

Frontend uses native browser APIs. Optional Google Fonts enhance typography; system fonts are the offline fallback. Book art and all model logic work without remote image/API dependencies. This app has no authentication or multi-user server storage and is intended for localhost demonstrations, not public production deployment.

## Research basis

The supplied project report defines the scope. This implementation follows its weighted relevance/novelty formulation; it does not claim to reproduce the exact algorithm from every cited paper. See Abdollahpouri et al. (2019), *Managing Popularity Bias in Recommender Systems with Personalized Re-ranking*, https://arxiv.org/abs/1901.07555, and Wan & McAuley (2018), https://doi.org/10.1145/3240323.3240369. Offline relevance is a proxy for preference, and increased tail exposure does not establish fairness or user satisfaction.
