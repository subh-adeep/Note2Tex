# author : Adithya
import pandas as pd
from subpart_split import parse_problem_text
import yaml
import os
import re

def save_problem_to_file(problem_obj, filename: str):
    if problem_obj is None:
        print(f"⚠️ Skipping {filename} — Result is None.")
        return

    # Check if the object is empty
    if not problem_obj.problems:
        print(f"⚠️ Warning: {filename} was saved but contains NO data (Regex failed).")

    # model_dump(mode='json') automatically handles the nested structure
    data = problem_obj.model_dump(mode='json')

    with open(filename, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data, 
            f, 
            sort_keys=False, 
            allow_unicode=True, 
            default_flow_style=False,
            width=1000
        )
    print(f"✅ Saved: {filename}")


def extract_questions_from_csv(file_path):
    df = pd.read_csv(file_path)
    df['latex_content'] = df['latex_content'].fillna('')
    df['question_start_type'] = df['question_start_type'].fillna('')

    def is_title_like(text: str) -> bool:
        t = text.strip()
        if len(t) <= 80:
            if re.search(r'^(##|#|\*\*)', t):
                return True
            if re.search(r'^Chapter\s+\d+', t, re.IGNORECASE):
                return True
            if re.search(r'^[A-Z][A-Za-z\s]+:$', t):
                return True
        return False

    groups = []
    current = []

    for _, row in df.iterrows():
        start_type = str(row['question_start_type']).strip().lower()
        text_segment = str(row['latex_content'])

        is_new = 'start of a new question' in start_type
        if is_new and not is_title_like(text_segment):
            if current:
                groups.append("\n".join(current))
                current = []
            current.append(text_segment)
        else:
            if text_segment.strip():
                current.append(text_segment)

    if current:
        groups.append("\n".join(current))

    def merge_short_titles(items, min_len=150):
        merged = []
        i = 0
        while i < len(items):
            g = items[i]
            if len(g.strip()) < min_len and i + 1 < len(items):
                merged.append(g + "\n" + items[i+1])
                i += 2
            else:
                merged.append(g)
                i += 1
        return merged

    return merge_short_titles(groups)

# -------------------------
# RUNNING THE EXTRACTION
# -------------------------

if __name__ == "__main__":
    candidates = [
        'qwen_debug_full_outputs.csv',
        os.path.join('classifiedBlocksOutput', 'qwen_debug_full_outputs.csv'),
    ]
    file_name = None
    for c in candidates:
        if os.path.exists(c):
            file_name = c
            break
    if not file_name:
        print("❌ Error: qwen_debug_full_outputs.csv not found in project root or classifiedBlocksOutput/.")
        exit()

    questions = extract_questions_from_csv(file_name)
    print(f"Successfully extracted {len(questions)} questions from {file_name}.")

    output_dir = 'yaml_parsed_questions'
    os.makedirs(output_dir, exist_ok=True)

    # Clean previous runs
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))

    # Process every question
    for i, question in enumerate(questions):
        print("="*40)
        print(f"\nProcessing question {i+1}/{len(questions)}...")
        
        if not question.strip():
            print("Skipping empty text block.")
            continue

        parsed = parse_problem_text(question)
        save_problem_to_file(parsed, f'{output_dir}/parsed_question_{i+1}.yaml')

    print("\nAll done.")
