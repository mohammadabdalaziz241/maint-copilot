# ViDoRe V3 Industrial: corpus check

## Licence (from the dataset card)

README.md (dataset card) not found in the download

## Files

| File | Size (MB) |
|---|---|
| `.cache/huggingface/.gitignore` | 0.0 |
| `.cache/huggingface/CACHEDIR.TAG` | 0.0 |
| `.cache/huggingface/download/corpus/test-00000-of-00005.parquet.metadata` | 0.0 |
| `.cache/huggingface/download/corpus/test-00001-of-00005.parquet.metadata` | 0.0 |
| `.cache/huggingface/download/corpus/test-00002-of-00005.parquet.metadata` | 0.0 |
| `.cache/huggingface/download/corpus/test-00003-of-00005.parquet.metadata` | 0.0 |
| `.cache/huggingface/download/corpus/test-00004-of-00005.parquet.metadata` | 0.0 |
| `.cache/huggingface/download/qrels/test-00000-of-00001.parquet.metadata` | 0.0 |
| `.cache/huggingface/download/queries/test-00000-of-00001.parquet.metadata` | 0.0 |
| `.cache/huggingface/trees/e26c864724f5dd71a3d7d739272d95637764cee9.json` | 0.0 |
| `corpus/test-00000-of-00005.parquet` | 295.5 |
| `corpus/test-00001-of-00005.parquet` | 435.8 |
| `corpus/test-00002-of-00005.parquet` | 449.1 |
| `corpus/test-00003-of-00005.parquet` | 456.1 |
| `corpus/test-00004-of-00005.parquet` | 414.9 |
| `qrels/test-00000-of-00001.parquet` | 0.1 |
| `queries/test-00000-of-00001.parquet` | 0.6 |

## Split: corpus

5244 rows from 5 file(s).

Schema: `corpus_id`: int64, `image`: struct<bytes: binary, path: string>, `doc_id`: string, `markdown`: string, `page_number_in_doc`: int64

## Split: queries

1698 rows from 1 file(s).

Schema: `query_id`: int64, `query`: string, `language`: string, `query_types`: list<element: string>, `query_format`: string, `content_type`: list<element: string>, `raw_answers`: list<element: string>, `query_generator`: string, `query_generation_pipeline`: string, `source_type`: string, `query_type_for_generation`: string, `answer`: string

## Split: qrels

9684 rows from 1 file(s).

Schema: `query_id`: int64, `corpus_id`: int64, `score`: int64, `content_type`: list<element: string>, `bounding_boxes`: list<element: struct<annotator: int64, x1: int64, x2: int64, y1: int64, y2: int64>>

### queries.language

| Value | Count |
|---|---|
| english | 283 |
| french | 283 |
| spanish | 283 |
| italian | 283 |
| german | 283 |
| portuguese | 283 |

### queries.query_types

| Value | Count |
|---|---|
| ['open-ended'] | 342 |
| ['extractive'] | 330 |
| ['boolean'] | 264 |
| ['enumerative' 'extractive'] | 198 |
| ['extractive' 'numerical'] | 138 |
| ['multi-hop'] | 102 |
| ['compare-contrast'] | 78 |
| ['compare-contrast' 'extractive'] | 72 |
| ['enumerative' 'multi-hop'] | 54 |
| ['enumerative' 'open-ended'] | 48 |
| ['enumerative'] | 18 |
| ['compare-contrast' 'open-ended'] | 12 |
| ['enumerative' 'extractive' 'numerical'] | 12 |
| ['numerical'] | 12 |
| ['compare-contrast' 'numerical'] | 6 |

### queries.query_format

| Value | Count |
|---|---|
| question | 966 |
| instruction | 504 |
| keyword | 228 |

Coverage below uses the 283 English queries only.

## Rotating-machinery coverage: queries

| Term group | queries matching | Share |
|---|---|---|
| bearing | 0 of 283 | 0.0% |
| vibration | 0 of 283 | 0.0% |
| rotating machinery | 0 of 283 | 0.0% |
| pump | 1 of 283 | 0.4% |
| turbine / engine | 8 of 283 | 2.8% |
| gear / gearbox | 0 of 283 | 0.0% |
| motor / generator | 2 of 283 | 0.7% |
| lubrication | 2 of 283 | 0.7% |
| fault / failure | 9 of 283 | 3.2% |
| inspection / maintenance | 25 of 283 | 8.8% |

### Queries mentioning bearing, vibration, rotating machinery (0)


## Documents (by `doc_id`)

27 distinct documents.

| Document | Pages |
|---|---|
| TO_00-25-259 | 892 |
| TO_32-1-101 | 490 |
| TO_1-1-691 | 410 |
| TO_36-1-191 | 383 |
| TO_1-1A-9 | 360 |
| TO_00-25-223 | 352 |
| TO_1-1-700 | 284 |
| TO_1-1-3 | 270 |
| AFD-100607-009 | 256 |
| TO_00-25-234 | 226 |
| TO_1-1-8 | 222 |
| TO_00-5-15 | 130 |
| TO_1-1-686_1 | 124 |
| TO_35-1-18 | 119 |
| TO_00-25-172 | 114 |
| TO_35-1-3 | 104 |
| TO_00-25-172CL-4 | 92 |
| TO_00-25-172CL-5 | 82 |
| TO_00-25-203 | 76 |
| TO_00-25-172CL-3 | 68 |
| TO_00-25-245_1 | 50 |
| TO_00-25-260 | 46 |
| TO_32B14-3-1-101 | 34 |
| TO_00-25-172CL-1 | 28 |
| AFD-091005-056 | 18 |
| TO_34-1-3 | 12 |
| AFD-091005-062 | 2 |

## Rotating-machinery coverage: corpus pages (`markdown`)

| Term group | pages matching | Share |
|---|---|---|
| bearing | 136 of 5244 | 2.6% |
| vibration | 80 of 5244 | 1.5% |
| rotating machinery | 180 of 5244 | 3.4% |
| pump | 264 of 5244 | 5.0% |
| turbine / engine | 438 of 5244 | 8.4% |
| gear / gearbox | 190 of 5244 | 3.6% |
| motor / generator | 188 of 5244 | 3.6% |
| lubrication | 772 of 5244 | 14.7% |
| fault / failure | 1236 of 5244 | 23.6% |
| inspection / maintenance | 1958 of 5244 | 37.3% |

### Documents with the most bearing/vibration/rotating-machinery pages

| Document | Matching pages |
|---|---|
| TO_36-1-191 | 58 |
| TO_32-1-101 | 52 |
| TO_1-1-691 | 43 |
| TO_1-1-700 | 37 |
| TO_1-1A-9 | 34 |
| TO_00-25-223 | 33 |
| TO_1-1-686_1 | 24 |
| TO_00-25-234 | 24 |
| TO_1-1-8 | 9 |
| TO_00-25-259 | 7 |
| TO_00-25-172 | 5 |
| TO_35-1-3 | 5 |
| TO_1-1-3 | 3 |
| TO_00-25-172CL-3 | 3 |
| TO_00-25-203 | 3 |
| TO_34-1-3 | 3 |
| AFD-100607-009 | 3 |
| TO_00-25-260 | 1 |
