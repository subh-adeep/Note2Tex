# author : Debanjan
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import re
import json
import os
import subprocess
import time
import random
import boto3
from tqdm.auto import tqdm
from botocore.exceptions import ClientError
import torch
env_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Notebook Processing", ".env")
if os.path.exists(env_file_path):
    with open(env_file_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#"): 
                continue
            if "=" in _line:
                _k, _v = _line.split("=", 1)
                _v = _v.strip().strip('"').strip("'")
                os.environ.setdefault(_k.strip(), _v)
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

# Set environment variable for Nougat
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
input_dir = "input_pdf/"
filename = "DIP_3.pdf"
# Define the PDF path
pdf_path = input_dir + filename

# AWS Bedrock Config
BEDROCK_REGION = "us-east-1"
BEDROCK_MODEL_ID = "meta.llama3-70b-instruct-v1:0"

# Check GPU Availability
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU Not Detected. Nougat will run on CPU (slower).")

# ==========================================
# 2. PDF TEXT EXTRACTION (PyMuPDF)
# ==========================================

records = []
try:
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found at {pdf_path}")
        # Create dummy file for testing flow if needed, or exit
    else:
        with fitz.open(pdf_path) as doc:
            for pno, page in enumerate(doc, start=1):
                for b in page.get_text("dict")["blocks"]:
                    if "lines" not in b: continue
                    for l in b["lines"]:
                        for s in l["spans"]:
                            t = s["text"].strip()
                            if not t: continue
                            x0,y0,x1,y1 = s["bbox"]
                            records.append({
                                "page": pno,
                                "text": t,
                                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                                "font": s["font"], "size": s["size"],
                                "bold": "Bold" in s["font"]
                            })
        df = pd.DataFrame(records)
        if not df.empty:
            df["mid_y"] = (df.y0 + df.y1)/2
            print(f"Extracted {len(df)} text spans.")
        else:
            print("⚠️ No text extracted from PDF.")

except Exception as e:
    print(f"Error opening or processing PDF: {e}")

# ==========================================
# 3. GROUPING & SECTION DETECTION
# ==========================================

def group_lines(df, y_gap=5):
    lines=[]
    for page, grp in df.groupby("page"):
        grp=grp.sort_values("y0")
        if grp.empty: continue
        buf=[grp.iloc[0]]
        for _,r in list(grp.iterrows())[1:]:
            if r.y0 - buf[-1].y1 < y_gap:
                buf.append(r)
            else:
                if buf: lines.append(buf); buf=[r]
        if buf: lines.append(buf)
    out=[]
    for ln in lines:
        if not ln: continue
        x0=min(l.x0 for l in ln); y0=min(l.y0 for l in ln)
        x1=max(l.x1 for l in ln); y1=max(l.y1 for l in ln)
        out.append({
            "page":ln[0].page,
            "bbox":[x0,y0,x1,y1],
            "text":" ".join(l.text for l in ln),
            "size":np.mean([l.size for l in ln]),
            "bold":any(l.bold for l in ln)
        })
    return out

def detect_sections(lines, gap_thresh=25, size_jump=1.4):
    sections=[]; curr=[]; last=None
    for i,l in enumerate(lines):
        new=False
        if last is not None:
            if l["page"]!=last["page"]:
                new=True
            else:
                gap = l["bbox"][1]-last["bbox"][3]
                if gap>gap_thresh: new=True
                elif last["size"] > 0 and l["size"] > 0 and l["size"]/last["size"]>size_jump: new=True
        
        if re.match(r"^\d+\.|^[A-Z].*?:", l["text"]) or l["bold"]:
            if last and not new:
                new=True
        
        if last and re.search(r"\bMarks?\b", last["text"], re.I):
            new=True

        if new and curr:
            x0=min(c["bbox"][0] for c in curr)
            y0=min(c["bbox"][1] for c in curr)
            x1=max(c["bbox"][2] for c in curr)
            y1=max(c["bbox"][3] for c in curr)
            
            sizes = [c["size"] for c in curr if c["size"] > 0]
            avg_size = np.mean(sizes) if sizes else 0.0

            sections.append({
                "id": len(sections),
                "page_start":curr[0]["page"],
                "page_end":curr[-1]["page"],
                "bbox":[x0,y0,x1,y1],
                "text":" ".join(c["text"] for c in curr),
                "bold": any(c["bold"] for c in curr),
                "size": avg_size
            })
            curr=[]
        curr.append(l); last=l

    if curr:
        x0=min(c["bbox"][0] for c in curr)
        y0=min(c["bbox"][1] for c in curr)
        x1=max(c["bbox"][2] for c in curr)
        y1=max(c["bbox"][3] for c in curr)
        
        sizes = [c["size"] for c in curr if c["size"] > 0]
        avg_size = np.mean(sizes) if sizes else 0.0

        sections.append({
            "id": len(sections),
            "page_start":curr[0]["page"],
            "page_end":curr[-1]["page"],
            "bbox":[x0,y0,x1,y1],
            "text":" ".join(c["text"] for c in curr),
            "bold": any(c["bold"] for c in curr),
            "size": avg_size
        })
    return sections

if 'df' in locals() and not df.empty:
    lines = group_lines(df)
    print(f"{len(lines)} merged text lines found.")
    sections = detect_sections(lines)
    print(f"{len(sections)} logical sections detected.")

    for s in sections:
        s["page_start"] = int(s["page_start"])
        s["page_end"] = int(s["page_end"])
        s["bbox"] = [float(x) for x in s["bbox"]]
        s["text"] = str(s["text"])
        s["bold"] = bool(s["bold"])
        s["size"] = float(s["size"])

    # Visualisation
    pdf_out = f"{filename.replace('.pdf', '')}_sections.pdf"
    if os.path.exists(pdf_path):
        doc_vis = fitz.open(pdf_path)
        for s in sections:
            rect = fitz.Rect(s["bbox"])
            if s["page_start"] <= len(doc_vis):
                page = doc_vis[s["page_start"] - 1]
                page.draw_rect(rect, color=(1, 0, 0), width=1.3)
        doc_vis.save(pdf_out)
        doc_vis.close()
        print(f"✅ Section visualization saved → {pdf_out}")
else:
    print("Skipping section detection.")

# ==========================================
# 4. NOUGAT PROCESSING
# ==========================================

def cleanup_mmd(mmd_text):
    if not isinstance(mmd_text, str): return ""
    latex = mmd_text.replace("$$", "\n\\[\n")
    latex = latex.replace("$", "\\(")
    latex = latex.replace("\\(\n\\[\n", "\\[")
    latex = latex.replace("\\]\n\\)", "\\]")
    return latex.strip()

def parse_sections_with_nougat(sections_list, original_pdf_path):
    print(f"Starting Nougat parsing for {len(sections_list)} sections.")
    
    # Ensure directories exist
    temp_pdf_dir = "temp_nougat_pdfs"
    mmd_dir = "nougat_output"
    os.makedirs(temp_pdf_dir, exist_ok=True)
    os.makedirs(mmd_dir, exist_ok=True)
    
    doc = fitz.open(original_pdf_path)
    parsed_sections = []

    for s in tqdm(sections_list):
        temp_pdf_path = None
        mmd_path = None
        try:
            page = doc[s["page_start"] - 1]
            rect = fitz.Rect(s["bbox"])
            page_width = page.rect.width
            clip_rect = fitz.Rect(0, rect.y0, page_width, rect.y1)

            temp_pdf_path = os.path.join(temp_pdf_dir, f"section_{s['id']}.pdf")
            mmd_path = os.path.join(mmd_dir, f"section_{s['id']}.mmd")

            temp_doc = fitz.open()
            temp_page = temp_doc.new_page(width=clip_rect.width, height=clip_rect.height)
            temp_page.show_pdf_page(temp_page.rect, doc, page.number, clip=clip_rect)
            temp_doc.save(temp_pdf_path)
            temp_doc.close()

            # Run Nougat
            # Added check to ensure we don't crash if nougat isn't installed
            try:
                env = os.environ.copy()
                if device == "cuda":
                    env["CUDA_VISIBLE_DEVICES"] = "0"
                
                subprocess.run(
                    ["nougat", temp_pdf_path, "-o", mmd_dir, "--no-skipping"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=True,
                    env=env
                )
                nougat_success = True
            except (FileNotFoundError, subprocess.CalledProcessError):
                nougat_success = False

            if nougat_success and os.path.exists(mmd_path):
                with open(mmd_path, "r", encoding="utf-8") as f:
                    s["latex_content"] = cleanup_mmd(f.read())
            else:
                s["latex_content"] = cleanup_mmd(s["text"])
            
            parsed_sections.append(s)

        except Exception as e:
            # If Nougat fails or isn't installed, fall back to simple text
            s["latex_content"] = cleanup_mmd(s["text"])
            parsed_sections.append(s)
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)
            if mmd_path and os.path.exists(mmd_path): os.remove(mmd_path)

    doc.close()
    return parsed_sections

if 'sections' in locals() and len(sections) > 0 and os.path.exists(pdf_path):
    sections_with_latex = parse_sections_with_nougat(sections, pdf_path)
    df_intermediate = pd.DataFrame(sections_with_latex)
else:
    if 'sections' in locals() and len(sections) > 0:
         # Fallback if PDF path issue but sections exist
         df_intermediate = pd.DataFrame(sections)
         df_intermediate['latex_content'] = df_intermediate['text']
    else:
        df_intermediate = pd.DataFrame()

# ==========================================
# 5. AWS BEDROCK INTEGRATION (Using Llama 3)
# ==========================================

print("\n=== Initializing AWS Bedrock ===")

def get_bedrock_client():
    try:
        return boto3.client(service_name="bedrock-runtime", region_name=BEDROCK_REGION)
    except Exception as e:
        print(f"❌ Failed to initialize Bedrock client: {e}")
        return None

def format_llama3_prompt(user_message: str, system_message: str) -> str:
    """
    Standard Llama 3 formatting tokens.
    """
    formatted = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_message}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return formatted

# --- RETRY LOGIC / EXPONENTIAL BACKOFF ---
def invoke_bedrock_with_backoff(client, body, model_id, max_attempts=8):
    """
    Invokes Bedrock with a retry loop to handle ThrottlingExceptions.
    """
    delay = 1.0 # Initial delay in seconds
    
    for attempt in range(max_attempts):
        try:
            response = client.invoke_model(
                body=json.dumps(body), 
                modelId=model_id,
                contentType="application/json",
                accept="application/json"
            )
            return response
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            
            # Check for Throttling
            if error_code == 'ThrottlingException':
                if attempt == max_attempts - 1:
                    print(f"❌ Bedrock Throttling: Max retries ({max_attempts}) reached.")
                    raise e
                
                # Exponential Backoff + Jitter
                sleep_time = delay + random.uniform(0, 0.5)
                # Optional: print(f"⚠️ Throttled. Retrying in {sleep_time:.2f}s... (Attempt {attempt+1}/{max_attempts})")
                time.sleep(sleep_time)
                delay *= 2 
            else:
                # If it's another error (e.g., ValidationException), raise immediately
                print(f"❌ Bedrock Client Error: {e}")
                raise e
        except Exception as e:
            print(f"❌ Unexpected Bedrock Error: {e}")
            raise e

def classify_block_with_bedrock(client, text, bold, size):
    """
    Uses AWS Bedrock (Llama 3 70B) to classify the block type.
    """
    if not client: return "Error"
    
    system_instruction = (
        "You are a document layout analyzer. "
        "Classify the text block into exactly ONE category: "
        "'Question_Start', 'Subpart', 'Instruction', or 'Solution'. "
        "Output ONLY the category name."
    )

    user_content = f"""
    BLOCK METADATA:
    - Text: "{text}"
    - Is Bold: {bold}
    - Font Size: {size}
    
    Task: Classify this block. Output ONLY the category name.
    """

    final_prompt = format_llama3_prompt(user_content, system_instruction)

    body = {
        "prompt": final_prompt,
        "max_gen_len": 20, # Keep it short for classification
        "temperature": 0.1,
        "top_p": 0.9,
    }

    try:
        # USE THE BACKOFF FUNCTION HERE
        response = invoke_bedrock_with_backoff(client, body, BEDROCK_MODEL_ID)
        
        response_body = json.loads(response.get("body").read())
        # Llama 3 returns text in 'generation' key
        result = response_body.get("generation").strip()
        return result
    except Exception as e:
        # print(f"Final Error classifying block: {e}")
        return "Unclassified"

print("=== Starting AWS Bedrock Classification (Llama 3) ===")

if not df_intermediate.empty:
    bedrock_client = get_bedrock_client()
    
    if bedrock_client:
        # We use tqdm to show progress as API calls take time
        tqdm.pandas(desc="Classifying with Bedrock")
        
        # Apply the function to the DataFrame
        df_intermediate['question_start_type'] = df_intermediate.progress_apply(
            lambda row: classify_block_with_bedrock(
                bedrock_client, 
                # Use latex_content if available, else raw text. limit chars to save tokens
                row.get('latex_content', row['text'])[:800], 
                row['bold'], 
                row['size']
            ), axis=1
        )
        print("✅ Classification complete.")
    else:
        print("Skipping classification due to client error.")
else:
    print("DataFrame empty, skipping Bedrock.")

# ==========================================
# 6. EXPORT
# ==========================================

import pickle

print("\n=== Exporting Final Data ===")

if not df_intermediate.empty:
    df_to_export = df_intermediate.copy().sort_values(by="id")
    export_dir = "classifiedBlocksOutput/"
    os.makedirs(export_dir, exist_ok=True)

    # Export to Pickle
    pkl_path = os.path.join(export_dir, "classified_blocks.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(df_to_export, f)
    
    # Export to CSV
    csv_path = os.path.join(export_dir, "qwen_debug_full_outputs.csv")
    try:
        df_to_export.to_csv(csv_path, index=False, escapechar='\\')
        print(f"✅ Saved to: {csv_path}")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")
    print(df_to_export[['id', 'latex_content', 'question_start_type']].head())
else:
    print("No data to export.")
