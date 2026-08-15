# Assembles the two things the Windows build actually needs from Vulkan --
# the headers and a GLSL compiler -- into a directory shaped like a Vulkan SDK
# root, and exports VULKAN_SDK pointing at it.
#
# Deliberately not the LunarG SDK installer: that is ~1 GB and several minutes
# per job, and the build links against nothing at all (the loader is resolved
# with LoadLibrary at runtime). Headers plus glslang is ~15 MB.
#
# Both come from Khronos' own repositories, pinned by tag rather than tracking
# latest, so a CI result is reproducible and an upstream retag cannot silently
# change what shipped.
#
# Shared by the CI and release workflows so their toolchains cannot drift.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # progress bars make Invoke-WebRequest crawl

$headersTag = $env:VULKAN_HEADERS_TAG
$glslangTag = $env:GLSLANG_TAG
if (-not $headersTag) { throw 'VULKAN_HEADERS_TAG not set' }
if (-not $glslangTag) { throw 'GLSLANG_TAG not set' }

$root = Join-Path $env:RUNNER_TEMP 'vulkan-build-deps'
$work = Join-Path $env:RUNNER_TEMP 'vulkan-dl'
New-Item -ItemType Directory -Force -Path $root, $work | Out-Null

# --- headers -------------------------------------------------------------
# Vulkan-Headers publishes no release assets, only tags, so take the tag
# archive. Its include/ becomes <root>\Include.
$headersZip = Join-Path $work 'vulkan-headers.zip'
curl.exe -L --fail --silent --show-error --retry 3 -o $headersZip `
  "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/$headersTag.zip"
if ($LASTEXITCODE -ne 0) { throw "downloading Vulkan-Headers $headersTag failed" }
Expand-Archive -Path $headersZip -DestinationPath $work -Force

$headersSrc = Join-Path $work "Vulkan-Headers-$headersTag\include"
if (-not (Test-Path $headersSrc)) { throw "unexpected Vulkan-Headers layout: $headersSrc missing" }
Copy-Item -Path $headersSrc -Destination (Join-Path $root 'Include') -Recurse -Force

# --- glslang -------------------------------------------------------------
# The Khronos release ships bin\glslang.exe and *not* glslangValidator.exe;
# gf16.vcxproj accepts either name. Only some tags carry Windows binaries, so
# the tag here cannot be bumped blindly -- check the release has assets.
$glslangZip = Join-Path $work 'glslang.zip'
curl.exe -L --fail --silent --show-error --retry 3 -o $glslangZip `
  "https://github.com/KhronosGroup/glslang/releases/download/$glslangTag/glslang-$glslangTag-windows-x86_64-release.zip"
if ($LASTEXITCODE -ne 0) { throw "downloading glslang $glslangTag failed" }
$glslangDir = Join-Path $work 'glslang'
Expand-Archive -Path $glslangZip -DestinationPath $glslangDir -Force
Copy-Item -Path (Join-Path $glslangDir 'bin') -Destination (Join-Path $root 'Bin') -Recurse -Force

# --- check the shape the projects look for -------------------------------
$header = Join-Path $root 'Include\vulkan\vulkan.h'
if (-not (Test-Path $header)) { throw "vulkan.h not found at $header" }

$compiler = Get-ChildItem -Path (Join-Path $root 'Bin') -Filter 'glslang*.exe' |
            Select-Object -First 1
if (-not $compiler) { throw "no glslang executable under $root\Bin" }

# Prove it runs before the build depends on it: a compiler that unpacked but
# cannot execute (missing runtime, wrong arch) would otherwise surface as a
# confusing build failure much later.
& $compiler.FullName --version
if ($LASTEXITCODE -ne 0) { throw "$($compiler.Name) did not run" }

"VULKAN_SDK=$root" | Out-File -FilePath $env:GITHUB_ENV -Append
Write-Host "VULKAN_SDK=$root  (headers $headersTag, glslang $glslangTag)"
