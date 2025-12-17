# GenAI Project

End‑to‑end pipeline to parse assignment PDFs, classify and link text blocks into questions, extract structured YAML, build a RAG index from notebook outputs, and generate concise LaTeX solutions (optionally compiling to PDF).

## Features
- PDF parsing and block detection with bounding boxes and visualization
- Block classification and question linking using Amazon Bedrock (Llama 3)
- YAML generation for problem → subproblem → sub‑subproblem hierarchy
- Optional RAG index over notebook outputs for context‑aware answers
- LaTeX solution generation and PDF compilation

## Prerequisites
- Python 3.10+
- AWS account with Bedrock access and credentials configured
- LaTeX toolchain (`pdflatex`) for PDF compilation (optional)
- GPU optional; Nougat will fall back to CPU if unavailable

## Setup
1. Create and activate a virtual/conda environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration
- Set AWS credentials via environment or the AWS CLI/SDK config (`~/.aws/credentials`).
- Region variables used by the code: `AWS_DEFAULT_REGION` or `BEDROCK_REGION` (default `us-east-1`).
- Do not store secrets in `.env`. Provide credentials securely via environment or IAM.

## Project Structure
- `input_pdf/` — source PDFs and section visualization output
- `export_for_kaggle/` — intermediate blocks for classification
- `classifiedBlocksOutput/` — final classified/linked blocks and CSV/PKL exports
- `yaml_parsed_questions/` — YAML files per parsed question
- `Jupyter file/` — parsed notebook JSON used to build the RAG index
- `RAG/` — FAISS index and utilities
- Scripts:
  - `preclassification.py` — parse PDF, detect sections, prepare blocks
  - `classifyAllBlocks.py` — classify blocks and link into questions
  - `extractQuestionFromCsv.py` — group linked blocks and produce YAML
  - `build_rag_index.py` — build FAISS index from notebook JSON
  - `generate_solution.py` — produce LaTeX solutions using RAG + Bedrock
  - `compile_pdf.py` — compile LaTeX to PDF with `pdflatex`

## Workflow
1. Preclassification
   ```bash
   python preclassification.py
   ```
   - Reads `input_pdf/DIP_3.pdf` (update path as needed)
   - Produces section visualization (`*_sections.pdf`)
   - Exports intermediate data to `classifiedBlocksOutput/` and/or `export_for_kaggle/`

2. Classify and Link Blocks
   ```bash
   python classifyAllBlocks.py
   ```
   - Saves `all_blocks_classified.{csv,pkl}`
   - Writes `qwen_debug_full_outputs.{csv,pkl}` containing “start of a new question” vs “continuation of previous question” labels

3. Extract Questions → YAML
   ```bash
   python extractQuestionFromCsv.py
   ```
   - Creates `yaml_parsed_questions/parsed_question_*.yaml`

4. (Optional) Build RAG Index
   ```bash
   python build_rag_index.py
   ```
   - Uses `Jupyter file/ans_parsed.json` to create `RAG/faiss_index`

5. Generate Solutions (LaTeX)
   ```bash
   python generate_solution.py
   ```
   - Reads YAML files in `yaml_parsed_questions/`
   - Uses Bedrock + RAG (if available) to draft concise LaTeX solutions
   - Writes `solution.tex`

6. Compile PDF (optional)
   ```bash
   python compile_pdf.py
   ```
   - Produces `solution.pdf` (requires `pdflatex` in PATH)

## Notes
- Bedrock models used include Llama 3 (`meta.llama3-70b-instruct-v1:0`).
- Nougat is invoked for math‑aware parsing where available; otherwise falls back to plain text.
- PowerShell helpers: `run.ps1`, `run_full_pipeline.ps1`; Bash helper: `run.sh`.

## Troubleshooting
- Credentials: verify `aws sts get-caller-identity` succeeds.
- Region: set `AWS_DEFAULT_REGION=us-east-1` if Bedrock client fails to initialize.
- LaTeX: install MiKTeX (Windows), MacTeX (macOS), or TeX Live (Linux).
