# Running the Solution Generator

## Prerequisites
Make sure you have the `genai` conda environment set up.

## Instructions

### 1. Activate the conda environment
```bash
conda activate genai
```

### 2. Run the solution generator
```bash
python generate_solution.py
```

### 3. Compile to PDF
```bash
python compile_pdf.py
```

## What the Script Does

The `generate_solution.py` script:

1. **Processes YAML files sequentially** (parsed_question_1.yaml, parsed_question_2.yaml, etc.)
2. **Checks the `answerable` field** - Only generates solutions for nodes where `answerable=true`
3. **Respects `output_format`**:
   - **`image`** or **`figure`**: Creates a placeholder image box with subtitle from the question content
   - **`text`** or **`answer`**: Generates LaTeX answer using RAG (vector search) + LLM
   - **`image+text`** or **`both`**: Creates placeholder image FIRST, then adds text answer below
4. **Generates LaTeX output** saved to `solution.tex`

The `compile_pdf.py` script:

1. **Compiles `solution.tex` to PDF** using pdflatex
2. **Runs twice** for proper references and table of contents
3. **Cleans up** auxiliary files (.aux, .log, etc.)
4. **Creates `solution.pdf`** in the project root

## Output
- **LaTeX File**: `solution.tex` (in the project root directory)
- **PDF File**: `solution.pdf` (compiled from solution.tex)

## LaTeX Requirements
To compile to PDF, you need LaTeX installed:
- **Windows**: Install [MiKTeX](https://miktex.org/download)
- **Linux**: `sudo apt-get install texlive-full`
- **Mac**: Install [MacTeX](https://www.tug.org/mactex/)

## Example YAML Structure
```yaml
problems:
  Problem 1:
    content: "Question text here"
    answerable: true          # ← Must be true to generate solution
    output_format: text       # ← Can be: text, image, image+text
    subproblems:
      (a):
        content: "Sub-question"
        answerable: true
        output_format: image+text  # ← Image placeholder + text explanation
```

## Image Placement
- Images use `[H]` (capital H) placement from the `float` package
- This ensures images appear **sequentially** right after their questions
- No more floating or out-of-order images!

