# Data Format

Retcon ingests `.txt`, `.md`, `.jsonl`, `.csv`, and `.parquet` sources.

For JSONL corpora, each line should contain at least:

```json
{"id":"doc-001","text":"Domain text goes here."}
```

Set `data_sources[].metadata.text_field` and `id_field` when your field names
differ. Evaluation JSONL files use the same surface format for perplexity, and
prompt/answer records for recall or application checks:

```json
{"id":"qa-001","prompt":"What fact matters?","answer":"The expected answer."}
```

Real private data belongs under ignored paths such as `data/source/domain/` and
`data/eval/`. Public examples live under `examples/`.
