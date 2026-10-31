# Data and licences

| Data | Licence | Status | How it is used |
|---|---|---|---|
| ViDoRe V3 Industrial (27 USAF Technical Orders) | Paper: "commercially permissive license"; documents selected to meet "open license criteria" (arXiv 2601.08620). All 27 manuals carry "Distribution Statement A. Approved for public release; distribution is unlimited." (checked in the page text, 31 Oct 2026). | Dataset licence: **CC BY 4.0** (Hugging Face dataset page, checked 31 Oct 2026) | Document corpus, public retrieval benchmark (nDCG@10), reference answers for end-to-end evaluation |
| `BAAI/bge-small-en-v1.5` embedding model | MIT (model card: "can be used for commercial purposes free of charge") | Read from the downloaded card, 31 Oct 2026 | Dense retrieval arm; weights downloaded to `models/`, not committed |

Rules: data stays in `data/` (git-ignored); the repository ships code, download scripts and result
files, not third-party documents or model weights.

## Attribution (CC BY 4.0)

This project uses the ViDoRe V3 Industrial dataset (`vidore/vidore_v3_industrial`), part of
ViDoRe V3: *A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World
Scenarios* (arXiv:2601.08620), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset is downloaded unmodified into
`data/` and is not redistributed in this repository. Derived artefacts that are committed (query-id
splits, per-query scores, extracted table schemas) are changes made by this project, are attributed
here, and are released under the same CC BY 4.0 terms. The underlying technical orders are U.S. Air
Force publications marked "Distribution Statement A. Approved for public release; distribution is
unlimited."
