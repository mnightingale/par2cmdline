#!/usr/bin/env pwsh
# Test 45: Repair damaged files in place (-i)

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

function Set-Bytes {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][int]$Offset,
        [Parameter(Mandatory=$true)][byte[]]$Bytes
    )

    $stream = [System.IO.File]::Open($Path, 'Open', 'Write')
    try {
        $stream.Seek($Offset, 'Begin') | Out-Null
        $stream.Write($Bytes, 0, $Bytes.Length)
    }
    finally {
        $stream.Dispose()
    }
}

try {
    Initialize-Test -TestName $testname

    # The flatdata fixture is long runs of a single byte value, so a great many
    # of its blocks are identical.  Corruption then desynchronises the sliding
    # window scan and the blocks after it are found at offsets that are not
    # their own, at which point repairing in place is no longer safe.  Use data
    # that does not repeat, so the test is about in-place repair rather than
    # about that.
    foreach ($f in @("test-1.data", "test-2.data", "test-3.data")) {
        New-RandomFile -Path $f -SizeBytes 204800
    }

    Write-Banner "create recovery files"

    Copy-Item "test-1.data" "test-1.data.orig"
    Copy-Item "test-2.data" "test-2.data.orig"

    # 60 blocks over three equal files is 20 blocks each, so 24 recovery blocks
    # is enough to rebuild a whole file from scratch, which one of the cases
    # below needs.
    $exitCode = Invoke-Par2 -Arguments @("c", "-b60", "-c24", "testdata.par2", "test-1.data", "test-2.data", "test-3.data")
    if ($exitCode -ne 0) {
        Exit-TestWithError "creating repair files using PAR 2.0 failed"
    }

    Write-Banner "repair a damaged file in place"

    # Corrupt 4 bytes in the middle of the file without changing its length.
    Set-Bytes -Path "test-1.data" -Offset 100000 -Bytes ([byte[]]@(0xde, 0xad, 0xbe, 0xef))

    $result = Invoke-Par2 -Arguments @("r", "-i", "testdata.par2") -ReturnObject
    Write-Host $result.StdOut
    if ($result.ExitCode -ne 0) {
        Exit-TestWithError "in-place repair using PAR 2.0 failed"
    }

    # Guard against a silent fallback making the rest of this pass for the wrong
    # reason.
    if ($result.StdOut -notmatch "in place") {
        Exit-TestWithError "the file was not repaired in place"
    }

    if (Test-Path "test-1.data.1") {
        Exit-TestWithError "in-place repair created a backup copy"
    }

    if (-not (Compare-Files "test-1.data" "test-1.data.orig")) {
        Exit-TestWithError "the in-place repaired file does not match the original"
    }

    $exitCode = Invoke-Par2 -Arguments @("v", "testdata.par2")
    if ($exitCode -ne 0) {
        Exit-TestWithError "verify after in-place repair failed"
    }

    Write-Banner "fall back to a new copy when the file is the wrong size"

    # The block positions in a file of the wrong length cannot be trusted, so
    # this one has to be repaired the usual way.
    $bytes = [System.IO.File]::ReadAllBytes("test-1.data.orig")
    [System.IO.File]::WriteAllBytes("test-1.data", $bytes[0..($bytes.Length - 2)])

    $result = Invoke-Par2 -Arguments @("r", "-i", "testdata.par2") -ReturnObject
    Write-Host $result.StdOut
    if ($result.ExitCode -ne 0) {
        Exit-TestWithError "fallback repair using PAR 2.0 failed"
    }
    if ($result.StdOut -notmatch "Cannot repair") {
        Exit-TestWithError "no reason was given for not repairing in place"
    }
    if (-not (Test-Path "test-1.data.1")) {
        Exit-TestWithError "fallback repair did not create a backup copy"
    }
    if (-not (Compare-Files "test-1.data" "test-1.data.orig")) {
        Exit-TestWithError "the repaired file does not match the original"
    }

    Remove-Item "test-1.data.1" -Force

    Write-Banner "do not repair in place when the file holds another file's data"

    # test-1.data's path now holds test-2.data's content, and test-2.data is
    # gone.  Overwriting test-1.data in place would destroy the only copy of
    # test-2.data.
    Copy-Item "test-2.data" "test-1.data" -Force
    Remove-Item "test-2.data" -Force

    $result = Invoke-Par2 -Arguments @("r", "-i", "testdata.par2") -ReturnObject
    Write-Host $result.StdOut
    if ($result.ExitCode -ne 0) {
        Exit-TestWithError "repair using PAR 2.0 failed"
    }
    if ($result.StdOut -notmatch "Cannot repair") {
        Exit-TestWithError "no reason was given for not repairing in place"
    }
    if (-not (Compare-Files "test-1.data" "test-1.data.orig")) {
        Exit-TestWithError "test-1.data does not match the original"
    }
    if (-not (Compare-Files "test-2.data" "test-2.data.orig")) {
        Exit-TestWithError "test-2.data was destroyed"
    }

    Remove-Item "test-1.data.1" -Force -ErrorAction SilentlyContinue
    Remove-Item "test-2.data.1" -Force -ErrorAction SilentlyContinue

    Write-Banner "default behaviour still creates a backup copy"

    Set-Bytes -Path "test-1.data" -Offset 100000 -Bytes ([byte[]]@(0xde, 0xad, 0xbe, 0xef))

    $exitCode = Invoke-Par2 -Arguments @("r", "testdata.par2")
    if ($exitCode -ne 0) {
        Exit-TestWithError "repair using PAR 2.0 failed"
    }
    if (-not (Test-Path "test-1.data.1")) {
        Exit-TestWithError "repair without -i stopped creating a backup copy"
    }
    if (-not (Compare-Files "test-1.data" "test-1.data.orig")) {
        Exit-TestWithError "the repaired file does not match the original"
    }

    Write-Banner "in-place repair and data skipping cannot be combined"

    $exitCode = Invoke-Par2 -Arguments @("r", "-i", "-N", "testdata.par2")
    if ($exitCode -eq 0) {
        Exit-TestWithError "-i -N was accepted"
    }
    $exitCode = Invoke-Par2 -Arguments @("r", "-N", "-i", "testdata.par2")
    if ($exitCode -eq 0) {
        Exit-TestWithError "-N -i was accepted"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
