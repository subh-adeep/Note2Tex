$ErrorActionPreference = 'Stop'

Write-Host '=================================================='
Write-Host 'GenAI Project: Full Automated Pipeline'
Write-Host '=================================================='

# Ensure we are in the script's directory
Set-Location -Path $PSScriptRoot

# 1. Run the existing PDF processing pipeline
Write-Host ''
Write-Host '[Step 1/3] Running PDF Processing & YAML Generation...'
Write-Host '--------------------------------------------------'
try {
    .\run.ps1
} catch {
    Write-Error "PDF Processing failed. Exiting."
    exit 1
}

# 2. Build the RAG Index
Write-Host ''
Write-Host '[Step 2/3] Building RAG Index...'
Write-Host '--------------------------------------------------'
python build_rag_index.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "RAG Index building failed. Exiting."
    exit $LASTEXITCODE
}

# 3. Generate the Solution
Write-Host ''
Write-Host '[Step 3/3] Generating Solution...'
Write-Host '--------------------------------------------------'
python generate_solution.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Solution generation failed. Exiting."
    exit $LASTEXITCODE
}

Write-Host ''
Write-Host '=================================================='
Write-Host '   ALL STEPS COMPLETED SUCCESSFULLY'
Write-Host '=================================================='
