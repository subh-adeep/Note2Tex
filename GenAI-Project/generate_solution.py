import os
import yaml
import json
import time
try:
    import boto3
except Exception:
    boto3 = None
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    FAISS = None
    HuggingFaceEmbeddings = None
from tqdm import tqdm

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_DIR = os.path.join(BASE_DIR, "yaml_parsed_questions")
RAG_INDEX_PATH = os.path.join(BASE_DIR, "RAG", "faiss_index")
OUTPUT_TEX_FILE = os.path.join(BASE_DIR, "solution.tex")

REGION_NAME = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

# Model ID - Using Llama 3 70B Instruct for high performance
# Alternative: "anthropic.claude-3-sonnet-20240229-v1:0"
MODEL_ID = "meta.llama3-70b-instruct-v1:0" 

def _load_env_credentials():
    paths = [
        os.path.join(BASE_DIR, ".env"),
        os.path.join(BASE_DIR, "Notebook Processing", ".env")
    ]
    for p in paths:
        if os.path.exists(p):
            if load_dotenv:
                load_dotenv(p, override=True)
            else:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            if "=" in line:
                                k, v = line.split("=", 1)
                                v = v.strip().strip('"').strip("'")
                                if k:
                                    os.environ.setdefault(k.strip(), v)
                except Exception:
                    pass
            break

def get_bedrock_client():
    if boto3 is None:
        return None
    try:
        ak = os.getenv("AWS_ACCESS_KEY_ID")
        sk = os.getenv("AWS_SECRET_ACCESS_KEY")
        st = os.getenv("AWS_SESSION_TOKEN")
        if ak and sk:
            session = boto3.Session(
                aws_access_key_id=ak,
                aws_secret_access_key=sk,
                aws_session_token=st,
                region_name=REGION_NAME,
            )
        else:
            session = boto3.Session(region_name=REGION_NAME)
        return session.client('bedrock-runtime')
    except Exception:
        return None

def load_rag_index(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"RAG index not found at {path}")
    if HuggingFaceEmbeddings is None or FAISS is None:
        print("RAG dependencies not available. Proceeding without context retrieval.")
        return None
    print(f"Loading RAG index from {path}...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vector_db

def generate_latex_with_bedrock(client, question_text: str, context_code: str) -> str:
    """
    Calls Amazon Bedrock to generate a CONCISE LaTeX solution.
    """
    
    # Strict System Prompt for Conciseness WITH Explanation
    system_prompt = """You are a LaTeX generation engine. 
    Input: A question and Python code context.
    Output: ONLY valid LaTeX code for the solution body.
    
    RULES:
    1. EXPLAIN the solution based on the provided code. Do not just write equations.
    2. Be CONCISE. Use clear, short sentences to explain the logic.
    3. NO conversational filler (e.g., "Here is the solution", "In this code"). Start directly with the explanation.
    4. NO \\documentclass, \\usepackage, or \\begin{document}. Just the content.
    5. Use standard LaTeX math formatting.
    6. If an image is needed, use a placeholder: \\begin{figure}[h] \\centering \\includegraphics[width=0.5\\textwidth]{placeholder.png} \\caption{...} \\end{figure}
    """

    user_message = f"""
    QUESTION: {question_text}
    
    CONTEXT CODE:
    {context_code}
    
    Generate the LaTeX solution with concise explanation now.
    """

    # Payload for Llama 3
    body = json.dumps({
        "prompt": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "max_gen_len": 2048,
        "temperature": 0.2,
        "top_p": 0.9
    })

    if client is None:
        return f"\\emph{{Solution generated offline.}}\\\n\\small{{{question_text}}}"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Rate limiting for Bedrock
            time.sleep(10) 
            
            response = client.invoke_model(
                body=body,
                modelId=MODEL_ID,
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            generation = response_body.get('generation')
            return generation.strip()

        except Exception as e:
            if "ThrottlingException" in str(e) and attempt < max_retries - 1:
                print(f"  ⚠️ Throttled. Retrying in 20s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(20)
            else:
                print(f"Bedrock API Error: {e}")
                return f"% Error generating solution: {e}"

def process_node(node, vector_db, bedrock_client, tex_file, labels=None, depth=0):
    """
    Recursively process the question tree and write to the tex file, with deterministic ordering and numbered headings.
    ONLY processes nodes where answerable=true
    """
    if labels is None:
        labels = []

    import re

    def _problem_num(k: str):
        m = re.search(r"Problem\s+(\d+)", k)
        return int(m.group(1)) if m else float('inf')

    def _alpha_order(k: str):
        m = re.search(r"\(?([a-zA-Z])\)?", k)
        ch = m.group(1).lower() if m else 'z'
        return ord(ch) - ord('a')

    def _roman_to_int(s: str):
        s = s.strip().lower().rstrip('.')
        vals = {'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100, 'd': 500, 'm': 1000}
        total = 0
        prev = 0
        for ch in reversed(s):
            val = vals.get(ch, 0)
            if val < prev:
                total -= val
            else:
                total += val
                prev = val
        return total if total > 0 else float('inf')

    # CHECK ANSWERABLE STATUS - Skip if answerable=false
    is_answerable = node.get("answerable", True)  # Default to True if not specified
    if not is_answerable:
        print(f"⏭️  Skipping (answerable=false): {node.get('content', 'No content')[:50]}...")
        return

    content = node.get("content")

    children_present = (
        (isinstance(node.get("problems"), dict) and len(node.get("problems")) > 0) or
        (isinstance(node.get("subproblems"), dict) and len(node.get("subproblems")) > 0) or
        (isinstance(node.get("sub_subproblems"), dict) and len(node.get("sub_subproblems")) > 0)
    )

    def sanitize_math(s: str) -> str:
        s = s.replace("π", r"\pi")
        s = s.replace("⋅", r"\cdot")
        s = s.replace("×", r"\times")
        s = s.replace("∀", r"\forall")
        s = s.replace("°", r"^{\circ}")
        return s

    if content:
        section_cmd = "\\section" if depth == 1 else ("\\subsection" if depth == 2 else ("\\subsubsection" if depth == 3 else "\\paragraph"))
        if depth == 1:
            header = " ".join(labels) if labels else "Question"
            tex_file.write(f"{section_cmd}{{{header}}}\n{content}\n\n")
        else:
            import re as _re
            eq_like = ("=" in content) or ("π" in content) or ("⋅" in content) or ("×" in content) or bool(_re.search(r"\b(sin|cos|tan|log|exp)\b", content))
            title = f"$ {sanitize_math(content)} $" if eq_like else content
            tex_file.write(f"{section_cmd}{{{title}}}\n\n")

        if not children_present:
            # Get output_format from YAML (preferred over output_type)
            output_format = node.get("output_format", node.get("output_type", ""))
            
            # Fallback: infer from content if not specified
            if not output_format:
                lc = content.lower()
                if any(w in lc for w in ["image", "plot", "spectrum", "figure", "magnitude spectrum"]):
                    output_format = "image"
                elif any(w in lc for w in ["explain", "comment", "report", "compute", "text"]):
                    output_format = "text"
                else:
                    output_format = "text"

            # Normalize
            output_format = output_format.strip().lower()
            print(f"📝 Processing ({output_format}): {content[:50]}...")

            tex_file.write("\\textbf{Solution:}\n\n")

            def _image_placeholder(subtitle=""):
                caption = subtitle if subtitle else "Placeholder for image output"
                tex_file.write(f"\\begin{{figure}}[H]\\n\\centering\\n\\fbox{{\\rule{{0pt}}{{2in}} \\rule{{0.8\\textwidth}}{{0pt}}}}\\n\\caption{{{caption}}}\\n\\end{{figure}}\\n\\n")

            # Handle different output formats
            if output_format in ("image", "figure"):
                # Image output - leave placeholder with subtitle
                subtitle = content[:100] if len(content) <= 100 else content[:97] + "..."
                _image_placeholder(subtitle)
                
            elif output_format in ("text", "answer"):
                # Text output - use RAG to generate answer
                docs = vector_db.similarity_search(content, k=1) if vector_db else []
                context_text = docs[0].page_content if docs else ""
                
                # DEBUG: Show RAG context retrieval
                if context_text:
                    print(f"   🔍 RAG Retrieved {len(context_text)} chars of context")
                    print(f"   📝 Context preview: {context_text[:100]}...")
                else:
                    print(f"   ⚠️  No RAG context found")
                
                latex_solution = generate_latex_with_bedrock(bedrock_client, content, context_text)
                tex_file.write(f"{latex_solution}\n\n")
                
            elif output_format in ("image+text", "text+image", "both"):
                # Both image and text
                subtitle = content[:100] if len(content) <= 100 else content[:97] + "..."
                _image_placeholder(subtitle)
                docs = vector_db.similarity_search(content, k=1) if vector_db else []
                context_text = docs[0].page_content if docs else ""
                
                # DEBUG: Show RAG context retrieval
                if context_text:
                    print(f"   🔍 RAG Retrieved {len(context_text)} chars of context")
                    print(f"   📝 Context preview: {context_text[:100]}...")
                else:
                    print(f"   ⚠️  No RAG context found")
                
                latex_solution = generate_latex_with_bedrock(bedrock_client, content, context_text)
                tex_file.write(f"{latex_solution}\n\n")
                
            else:
                # Default to text with RAG
                docs = vector_db.similarity_search(content, k=1) if vector_db else []
                context_text = docs[0].page_content if docs else ""
                latex_solution = generate_latex_with_bedrock(bedrock_client, content, context_text)
                tex_file.write(f"{latex_solution}\n\n")

        tex_file.write("\\hrule\\vspace{0.5cm}\n\n")

    # Process children in order
    if "problems" in node and isinstance(node["problems"], dict):
        for k in sorted(node["problems"].keys(), key=_problem_num):
            process_node(node["problems"][k], vector_db, bedrock_client, tex_file, labels=labels + [k], depth=depth + 1)

    if "subproblems" in node and isinstance(node["subproblems"], dict):
        for k in sorted(node["subproblems"].keys(), key=_alpha_order):
            process_node(node["subproblems"][k], vector_db, bedrock_client, tex_file, labels=labels + [k], depth=depth + 1)

    if "sub_subproblems" in node and isinstance(node["sub_subproblems"], dict):
        for k in sorted(node["sub_subproblems"].keys(), key=_roman_to_int):
            process_node(node["sub_subproblems"][k], vector_db, bedrock_client, tex_file, labels=labels + [k], depth=depth + 1)

def main():
    print("Initializing Amazon Bedrock Client...")
    bedrock_client = None
    try:
        _load_env_credentials()
        bedrock_client = get_bedrock_client()
        if bedrock_client is not None:
            print("✅ Bedrock client initialized")
        else:
            print("Proceeding without Bedrock. Solutions will use offline placeholders.")
    except Exception as e:
        print(f"Failed to connect to Bedrock: {e}")
        bedrock_client = None

    try:
        vector_db = load_rag_index(RAG_INDEX_PATH)
        print("✅ RAG index loaded successfully")
    except Exception as e:
        print(f"Failed to load RAG: {e}")
        return

    if not os.path.exists(QUESTIONS_DIR):
        print(f"Questions directory not found: {QUESTIONS_DIR}")
        return

    # Get all YAML files and sort them numerically
    import re
    yaml_files = [f for f in os.listdir(QUESTIONS_DIR) if f.endswith('.yaml') or f.endswith('.yml')]
    
    def extract_number(filename):
        """Extract number from filename like 'parsed_question_1.yaml' -> 1"""
        match = re.search(r'_(\d+)\.', filename)
        return int(match.group(1)) if match else float('inf')
    
    yaml_files.sort(key=extract_number)
    
    if not yaml_files:
        print(f"No YAML files found in {QUESTIONS_DIR}")
        return
    
    print(f"\n📁 Found {len(yaml_files)} YAML file(s): {', '.join(yaml_files)}")
    print(f"{'='*60}\n")

    with open(OUTPUT_TEX_FILE, "w", encoding="utf-8") as tex_file:
        # Write Header
        tex_file.write(r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{float}
\usepackage{geometry}
\geometry{a4paper, margin=1in}

\begin{document}
\title{Assignment Solutions}
\author{AI Assistant}
\date{\today}
\maketitle

""")
        
        # Process each YAML file sequentially
        for yaml_file in yaml_files:
            f_path = os.path.join(QUESTIONS_DIR, yaml_file)
            print(f"\n{'='*60}")
            print(f"📄 Processing: {yaml_file}")
            print(f"{'='*60}")
            
            try:
                with open(f_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if data:
                    print(f"✓ Loaded data keys: {list(data.keys())}")
                    process_node(data, vector_db, bedrock_client, tex_file, labels=[])
                else:
                    print(f"⚠️  Warning: No data in {yaml_file}")
            except Exception as e:
                print(f"❌ Error processing {yaml_file}: {e}")
                tex_file.write(f"\n% Error processing {yaml_file}: {e}\n\n")

        # Write Footer
        tex_file.write(r"\end{document}")

    print(f"\n{'='*60}")
    print(f"✅ Solution generated: {OUTPUT_TEX_FILE}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

