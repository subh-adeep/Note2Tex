import os
import subprocess
import sys

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEX_FILE = os.path.join(BASE_DIR, "solution.tex")
PDF_FILE = os.path.join(BASE_DIR, "solution.pdf")

def compile_tex_to_pdf():
    """
    Compile solution.tex to PDF using pdflatex
    """
    if not os.path.exists(TEX_FILE):
        print(f"❌ Error: {TEX_FILE} not found!")
        print(f"Please run 'python generate_solution.py' first to generate the LaTeX file.")
        return False
    
    print(f"{'='*60}")
    print(f"📄 Compiling LaTeX to PDF")
    print(f"{'='*60}\n")
    print(f"Input:  {TEX_FILE}")
    print(f"Output: {PDF_FILE}\n")
    
    # Check if pdflatex is available
    try:
        result = subprocess.run(
            ["pdflatex", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            raise Exception("pdflatex not found")
    except Exception as e:
        print(f"❌ Error: pdflatex is not installed or not in PATH")
        print(f"\nTo install LaTeX:")
        print(f"  - Windows: Install MiKTeX from https://miktex.org/download")
        print(f"  - Linux:   sudo apt-get install texlive-full")
        print(f"  - Mac:     Install MacTeX from https://www.tug.org/mactex/")
        return False
    
    # Run pdflatex twice (for references and table of contents)
    for run_num in range(1, 3):
        print(f"🔄 Running pdflatex (pass {run_num}/2)...")
        try:
            result = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",  # Don't stop on errors
                    "-output-directory=" + BASE_DIR,
                    TEX_FILE
                ],
                capture_output=True,
                text=True,
                cwd=BASE_DIR,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"⚠️  Warning: pdflatex returned error code {result.returncode}")
                print(f"\nShowing last 30 lines of output:")
                print("-" * 60)
                output_lines = result.stdout.split('\n')
                for line in output_lines[-30:]:
                    print(line)
                print("-" * 60)
            else:
                print(f"✅ Pass {run_num} completed successfully")
                
        except subprocess.TimeoutExpired:
            print(f"❌ Error: Compilation timed out (>60 seconds)")
            return False
        except Exception as e:
            print(f"❌ Error during compilation: {e}")
            return False
    
    # Check if PDF was created
    if os.path.exists(PDF_FILE):
        file_size = os.path.getsize(PDF_FILE) / 1024  # KB
        print(f"\n{'='*60}")
        print(f"✅ PDF successfully created!")
        print(f"{'='*60}")
        print(f"📍 Location: {PDF_FILE}")
        print(f"📊 Size: {file_size:.2f} KB")
        print(f"\nYou can now open the PDF file.")
        
        # Clean up auxiliary files
        cleanup_extensions = ['.aux', '.log', '.out', '.toc']
        print(f"\n🧹 Cleaning up auxiliary files...")
        for ext in cleanup_extensions:
            aux_file = os.path.join(BASE_DIR, f"solution{ext}")
            if os.path.exists(aux_file):
                try:
                    os.remove(aux_file)
                    print(f"  Removed: solution{ext}")
                except:
                    pass
        
        return True
    else:
        print(f"\n❌ Error: PDF file was not created")
        print(f"Check the LaTeX log file for details: solution.log")
        return False

if __name__ == "__main__":
    success = compile_tex_to_pdf()
    sys.exit(0 if success else 1)
