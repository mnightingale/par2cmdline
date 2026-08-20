#!/usr/bin/env pwsh
# Test 46: Checking the hash of the whole of a file when asked to

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "full-hash-mismatch.tar.gz") -Destination "."

    Write-Banner "Checking the hash of the whole of a file when asked to"

    # Every block of the file matches its verification entry, so the blocks
    # alone say the file is intact.
    $blocks = Invoke-Par2 -Arguments @("v", "-q", "recovery.par2") -ReturnObject
    if ($blocks.ExitCode -ne 0) {
        Exit-TestWithError "Expected the file to verify when only its blocks are checked"
    }

    # The whole file hash recorded for it does not match, which only shows up
    # when the whole of the file is hashed as well.
    $whole = Invoke-Par2 -Arguments @("v", "-q", "--force-full-hash-verify", "recovery.par2") -ReturnObject
    if ($whole.ExitCode -eq 0) {
        Exit-TestWithError "--force-full-hash-verify did not notice the whole file hash"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
