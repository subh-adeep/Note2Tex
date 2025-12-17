$ErrorActionPreference = 'Stop'
Write-Host '==========================='
Write-Host 'GenAI Project: Full Pipeline (Windows)'
Write-Host '==========================='
Set-Location -Path $PSScriptRoot
function Invoke-Step {
    param(
        [string]$Title,
        [string]$Script
    )
    Write-Host '----------------------------------------'
    Write-Host $Title
    Write-Host '----------------------------------------'
    python $Script
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Step failed: $Script (exit code $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error 'Python is not available on PATH. Activate your conda env first.'
    exit 1
}
Invoke-Step -Title '[1/3] Running preclassification.py ...' -Script 'preclassification.py'
Write-Host '✓ Output stored in export_for_kaggle/ and classifiedBlocksOutput/'
Invoke-Step -Title '[2/3] Running classifyAllBlocks.py ...' -Script 'classifyAllBlocks.py'
Write-Host '✓ Output stored in all_blocks_classified.* and qwen_debug_full_outputs.*'
Invoke-Step -Title '[3/3] Running extractQuestionFromCsv.py ...' -Script 'extractQuestionFromCsv.py'
Write-Host '✓ YAML outputs stored in yaml_parsed_questions/'
Write-Host ''
Write-Host '========================================='
Write-Host '   FULL PIPELINE EXECUTED SUCCESSFULLY'
Write-Host '========================================='
