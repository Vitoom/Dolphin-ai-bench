# Deep Scan: Report Generation Evaluation Pipeline

## Overview

This document presents a targeted, exhaustive analysis of the Dolphin-ai-bench (u2-bench-evalkit) codebase, focusing exclusively on the evaluation pipeline for generated ultrasound reports. The scan covers 4 areas: NLG/Clinical metrics, medical entity extraction, LLM-as-a-Judge, and the benchmarking pipeline entry point.

---

## 1. Traditional NLG & Clinical Metrics

**The only report evaluation metrics implemented are BLEU, ROUGE, and BERTScore.** There are no clinical-specific metrics whatsoever.

- **Primary file:** `new_results/qwen_report_eval.py` (346 lines) — the production version
- **Legacy file:** `new_results/report_eval.py` (246 lines) — earlier, simpler variant

### Core Scoring Class — `MedicalReportScorer`

```python
# qwen_report_eval.py:105-157
class MedicalReportScorer:
    def __init__(self, refs, hyps):
        self.refs = refs  # dict: {case_id: [reference_text]}
        self.hyps = hyps  # dict: {case_id: [generated_text]}
        self.scorers = [
            (Bleu(4), 'Bleu'),       # pycocoevalcap.bleu.bleu.Bleu
            (Rouge(), 'Rouge')        # pycocoevalcap.rouge.rouge.Rouge
        ]

    def evaluate(self):
        metrics = {}
        for scorer, name in self.scorers:
            score, scores = scorer.compute_score(self.refs, self.hyps)
            if name == 'Bleu':
                for i, v in enumerate(score):
                    metrics[f'Bleu-{i+1}'] = v * 100  # percentage scale
            else:
                metrics[name] = score * 100

        # BERTScore (F1 component)
        bert_scores = batch_bertscore(self.refs, self.hyps)
        metrics['BERTScore'] = np.mean(list(bert_scores.values())) * 100
        return metrics
```

### BERTScore Implementation — Singleton Model Cache + Batch Computation

```python
# qwen_report_eval.py:22-60
class BertModelCache:
    """Singleton pattern for BERT model (avoids reloading per evaluation)"""
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = BertModelCache()
        return cls._instance

    def load_bert_model(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        return tokenizer, model, device
```

```python
# qwen_report_eval.py:62-103
def batch_bertscore(refs_dict, hyps_dict):
    cache = BertModelCache.instance()
    all_hyp, all_ref = [], []
    for case_id, hyp in hyps_dict.items():
        ref = refs_dict.get(case_id, [])
        if ref:
            all_hyp.extend(hyp)
            all_ref.extend(ref)

    if len(all_hyp) > 0:
        _, _, F1 = bert_score.score(
            all_hyp, all_ref,
            model_type="bert-base-multilingual-cased",
            lang="en", device=cache.device, batch_size=8
        )
        # Aggregate F1 per case
        idx = 0
        for case_id, hyp in hyps_dict.items():
            refs = refs_dict.get(case_id, [])
            if refs:
                scores[case_id] = F1[idx:idx+len(refs)].mean().item()
                idx += len(refs)
    return scores
```

### What is NOT Implemented (Confirmed Zero Matches Across Entire Repo)

| Metric | Present? |
|---|---|
| METEOR | **No** |
| CIDEr | **No** |
| Clinical Efficacy (CE) | **No** |
| RadGraph F1 | **No** |
| Medical entity F1 | **No** |
| Factuality scoring | **No** |
| Hallucination detection | **No** |

---

## 2. Medical Entity Extraction & Alignment

**There is no medical entity extraction for report evaluation.** The report evaluation pipeline performs zero structural analysis of clinical content — it treats generated reports as opaque strings and computes only surface-level n-gram overlap (BLEU/ROUGE) and embedding similarity (BERTScore).

The closest thing to "extraction" in the codebase is in **non-report tasks**:

### Measurement Value Extraction (Not Report-Related)

```python
# measure_eval.py:42-60 — Parses JSON measurement strings per task
def extract_mea(mea_str, task_id):
    data = json.loads(mea_str)
    if task_id == '18':   return data.get('EF', data)           # Ejection Fraction
    elif task_id == '27': return list(data.get('IMT', data))[0] # Intima-Media Thickness
    elif task_id == '31': return data                           # Full thyroid dict
    elif task_id == '50': return data.get('abdominal_circumference', data)
    else:                 return data.get('fat value', data)    # Liver fat
```

### Classification Answer Extraction (Not Report-Related)

```python
# cla_eval.py:49-76 — Special anatomy task: extracts from nested JSON
classes_data = json.loads(row['classes'])
if isinstance(classes_data, dict):
    cla_ans = classes_data.get('Anatomy', '')
elif isinstance(classes_data, list):
    for item in classes_data:
        if isinstance(item, dict) and 'Anatomy' in item:
            cla_ans = item['Anatomy']
```

### Bounding Box to Location Label (Segmentation Only)

```python
# seg_eval.py:77-117
def get_bounding_box_location_v1(gt_bbox):
    """Converts (center_x, center_y) -> categorical position string"""
    LOW_THRESHOLD = 0.45
    HIGH_THRESHOLD = 0.55
    # -> "upper left", "center", "lower right", etc.
```

**Confirmed absent:** No spaCy, Stanza, SciSpaCy, regex-based clinical NER, or LLM-based entity extraction anywhere in the codebase.

---

## 3. LLM-as-a-Judge Implementation

**There is no LLM-as-a-Judge implementation anywhere in the repository.**

The exhaustive search confirmed:

| Pattern Searched | Result |
|---|---|
| `judge` / `Judge` | Only as `**judge_kwargs` parameter name — used solely for numerical `tolerance` threshold |
| `hallucination` | **0 matches** |
| `factuality` | **0 matches** |
| `prompt.*template` | **0 matches** |
| `system.*prompt` | **0 matches** |
| `LLM.*judge` | **0 matches** |
| `gpt.*eval` | **0 matches** |

The misleadingly-named `judge_kwargs` in `measure_eval.py`:

```python
# measure_eval.py:83 — This is NOT an LLM judge
def meaeval(data: pd.DataFrame, task_id: str, jsonl_path: str = None, **judge_kwargs) -> dict:
    tolerance = judge_kwargs.get('tolerance', 0.1)  # purely numerical threshold
```

All evaluation is **rule-based and deterministic**:

- **Classification/Segmentation:** case-insensitive substring matching
- **Measurement:** numerical error metrics (RMSE, MAE) with min-max normalization
- **Report:** n-gram overlap (BLEU/ROUGE) + embedding similarity (BERTScore)

---

## 4. Benchmarking Pipeline Entry Point

### Report Evaluation Entry Point

```python
# qwen_report_eval.py:273-315
def batch_evaluate_all_tasks(base_dir='data/report', output_file='results/report_results.txt'):
    all_results = []
    for task_id in TSV_PATH:                              # iterates '10', '11', '39'
        task_pairs = find_model_files(base_dir, task_id)  # discover JSONL files
        for jsonl_path, tsv_path in task_pairs:
            data = read_jsonl_with_tsv(jsonl_path, tsv_path)  # merge predictions + ground truth
            metrics = model_eval(data, jsonl_path=jsonl_path)  # compute BLEU/ROUGE/BERTScore
            all_results.append({
                'task_id': task_id, 'model': model_name,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **metrics
            })
    save_results(all_results, output_file)

if __name__ == "__main__":
    batch_evaluate_all_tasks()
```

### Data Flow: JSONL + TSV Merge

```python
# qwen_report_eval.py:185-215
def read_jsonl_with_tsv(jsonl_path, tsv_path):
    """Reads model predictions (JSONL) and merges with ground-truth captions (TSV)"""
    with open(jsonl_path, 'r') as f:
        jsonl_data = [json.loads(line) for line in f]
    df_jsonl = pd.DataFrame(jsonl_data)          # columns: id, response, ...
    df_tsv = pd.read_csv(tsv_path, sep='\t')     # columns: caption, ...
    # Length mismatch handling
    if len(df_jsonl) != len(df_tsv):
        min_len = min(len(df_jsonl), len(df_tsv))
        df_jsonl = df_jsonl.iloc[:min_len]
        df_tsv = df_tsv.iloc[:min_len]
    df_merged = df_jsonl.assign(cap_ans=df_tsv['caption'])
    return df_merged
```

### Task-to-Dataset Mapping (Report Tasks Only)

```python
# qwen_report_eval.py:15-20
TSV_PATH = {
    '10': '/media/ps/data-ssd/json_processing/ale_tsv_output/10.tsv',
    '11': '/media/ps/data-ssd/json_processing/ale_tsv_output/11.Thyroid_US_Images.tsv',
    '39': '/media/ps/data-ssd/json_processing/ale_tsv_output/39_translated.tsv',
}
```

### Output Format

```python
# qwen_report_eval.py:317-342
def save_results(results, output_file):
    columns = ['task_id', 'model', 'timestamp',
               'Bleu-1', 'Bleu-2', 'Bleu-3', 'Bleu-4',
               'Rouge', 'BERTScore']
    # Writes TSV with values formatted to 4 decimal places
```

### Final Leaderboard Aggregation

```python
# table_fill/update_table_score_format.py — Aggregates all 4 task types with weights
TASK_WEIGHTS = {'diag': 0.2, 'od': 0.27, 'mea': 0.23, 'report': 0.3}
# Reads 4 result TSVs -> computes weighted scores -> outputs LaTeX table
```

---

## Summary & Honest Assessment

The Dolphin-ai-bench report evaluation pipeline is **minimal by design** — it uses only 3 standard NLG metrics (BLEU-1/2/3/4, ROUGE, BERTScore) with no clinical-domain adaptations.

| Capability | Status |
|---|---|
| BLEU / ROUGE | Implemented via `pycocoevalcap` |
| BERTScore | Implemented via `bert-score` + `bert-base-multilingual-cased` |
| METEOR / CIDEr | Not implemented |
| Clinical entity extraction | Not implemented |
| Factuality / hallucination checking | Not implemented |
| LLM-as-a-Judge | Not implemented |
| RadGraph / Clinical Efficacy | Not implemented |
| spaCy / Stanza / SciSpaCy NER | Not used |

---

## References for Building a Clinical Factuality Pipeline

If building a clinical factuality pipeline for ultrasound report evaluation from scratch, established references include:

- **RadGraph** (Stanford) — radiology entity graph extraction
- **F1-RadGraph** — report-level factual correctness scoring
- **CheXbert** — clinical efficacy (CE) label extraction from radiology reports
- **GREEN** (Ostmeier et al.) — LLM-as-judge radiology report evaluation
- **FineRadScore** — fine-grained error annotation via LLM judges
