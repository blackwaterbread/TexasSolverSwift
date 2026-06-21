<#
.SYNOPSIS
  Benchmark: CPU (TexasSolverConsole, OpenMP) vs GPU (river_gpu, CUDA) solve speed.

.DESCRIPTION
  For each test spot we measure pure CFR-iteration throughput (iters/s) on both
  engines solving the SAME subgame, then report the GPU speedup.

  Fairness / methodology:
    * GPU: river_gpu self-reports the train()-loop wall-clock ("solved N in X s").
      That is exactly the per-iteration CFR compute (kernel launches / graph replay).
    * CPU: the console solver's qDebug timer can't be captured (Qt writes straight to
      the console), so we time the WHOLE process at two iteration counts (lo, hi) and
      difference them:  cpu_iter_time = (wall(hi) - wall(lo)) / (hi - lo).
      The constant overhead (compairer load, tree build, EV pass, json dump) is the
      same in both runs and cancels, leaving pure iteration time. print_interval is
      set huge so the periodic best-response pass never fires.
    * Both run with isomorphism OFF (the GPU engine has no isomorphism) -> an
      apples-to-apples per-iteration comparison. CPU thread count is per the spot
      config (reported per row). Per-spot (lo,hi) keep each run a few seconds.

.PARAMETER Spots     Optional subset of spot names to run (default: all).
.PARAMETER Scale     Multiplies every spot's (lo,hi) iteration counts (default 1.0;
                     raise for steadier figures, lower for a quick pass).
#>
param(
    [string[]]$Spots = @(),
    [double]$Scale = 1.0
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repo

# Qt/MinGW runtime DLLs for the CPU console exe + SerializeRiver.
$env:PATH = "C:\Qt\5.15.2\mingw81_64\bin;C:\Qt\Tools\mingw810_64\bin;" + $env:PATH

$cpuExe = Join-Path $repo "cuda\cpu_export\build\release\TexasSolverConsole.exe"
$gpuExe = Join-Path $repo "cuda\build\river_gpu.exe"
$resDir = Join-Path $repo "resources"
foreach ($e in @($cpuExe, $gpuExe)) {
    if (-not (Test-Path $e)) { throw "missing executable: $e" }
}

# name -> (cpu config, gpu serialized subgame, lo/hi iteration counts for differencing)
$allSpots = @(
    @{ name = "river";      cfg = "cuda\configs\river_spot.txt";      sg = "cuda\configs\subgame.txt";           lo = 50; hi = 1050 },
    @{ name = "river_big";  cfg = "cuda\configs\river_big_spot.txt";  sg = "cuda\configs\subgame_river_big.txt"; lo = 50; hi = 1050 },
    @{ name = "river_wide"; cfg = "cuda\configs\river_wide_spot.txt"; sg = "cuda\configs\subgame_river_wide.txt"; lo = 50; hi = 1050 },
    @{ name = "turn";       cfg = "cuda\configs\turn_spot.txt";       sg = "cuda\configs\subgame_turn.txt";       lo = 40; hi = 240 },
    @{ name = "flop_mix";   cfg = "cuda\configs\flop_mix_spot.txt";   sg = "cuda\configs\subgame_flop_mix.txt";   lo = 20; hi = 120 }
)
if ($Spots.Count -gt 0) { $allSpots = $allSpots | Where-Object { $Spots -contains $_.name } }

# Write a temp CPU config: override iterations, suppress the periodic BR pass
# (print_interval huge), and redirect the dump so the goldens are never clobbered.
function New-CpuConfig($srcRel, $iters, $dumpPath) {
    $src = Join-Path $repo $srcRel
    $lines = Get-Content $src
    $out = New-Object System.Collections.Generic.List[string]
    foreach ($ln in $lines) {
        if     ($ln -match '^\s*set_max_iteration')  { $out.Add("set_max_iteration $iters") }
        elseif ($ln -match '^\s*set_print_interval') { $out.Add("set_print_interval 99999999") }
        elseif ($ln -match '^\s*set_accuracy')       { $out.Add("set_accuracy 0") }
        elseif ($ln -match '^\s*dump_result')        { $out.Add("dump_result $dumpPath") }
        else { $out.Add($ln) }
    }
    $tmp = Join-Path $env:TEMP ("bench_cpu_{0}_{1}.txt" -f ([IO.Path]::GetFileNameWithoutExtension($srcRel)), $iters)
    Set-Content -Path $tmp -Value $out -Encoding ASCII
    return $tmp
}

function Get-ThreadNum($srcRel) {
    $m = (Get-Content (Join-Path $repo $srcRel) | Select-String '^\s*set_thread_num\s+(\d+)')
    if ($m) { return [int]$m.Matches[0].Groups[1].Value } else { return 1 }
}

# Wall-clock seconds for one full CPU solve process.
function Measure-CpuRun($cfgRel, $iters) {
    $dump = Join-Path $env:TEMP "bench_cpu_dump.json"
    $tmp = New-CpuConfig $cfgRel $iters $dump
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $cpuExe -i $tmp -r $resDir *> $null
    $sw.Stop()
    return $sw.Elapsed.TotalSeconds
}

# CPU iters/s via two-point differencing (cancels fixed load/build/dump overhead).
function Measure-CpuIps($cfgRel, $lo, $hi) {
    $tLo = Measure-CpuRun $cfgRel $lo
    $tHi = Measure-CpuRun $cfgRel $hi
    $dt = $tHi - $tLo
    if ($dt -le 0) { return [double]::NaN }
    return ($hi - $lo) / $dt
}

$rows = @()
foreach ($spot in $allSpots) {
    if (-not (Test-Path (Join-Path $repo $spot.sg))) {
        Write-Host ("[skip] {0}: missing subgame {1}" -f $spot.name, $spot.sg) -ForegroundColor Yellow
        continue
    }
    $threads = Get-ThreadNum $spot.cfg
    $lo = [int][math]::Ceiling($spot.lo * $Scale)
    $hi = [int][math]::Ceiling($spot.hi * $Scale)
    Write-Host ("=== {0} (CPU threads={1}) ===" -f $spot.name, $threads) -ForegroundColor Cyan

    # ---- CPU: two-point process-time differencing ----
    $cpuIps = Measure-CpuIps $spot.cfg $lo $hi
    Write-Host ("  CPU: {0}->{1} iters diff  ->  {2:N1} iters/s" -f $lo, $hi, $cpuIps)

    # ---- GPU: self-reported train()-loop throughput ----
    $dump = Join-Path $env:TEMP "bench_gpu_dump.json"
    $gpuOut = & $gpuExe -s (Join-Path $repo $spot.sg) -d $dump -n $hi 2>&1
    $line = ($gpuOut | Select-String 'solved .* iters/s').ToString()
    $gpuIps = [double]::NaN; $gpuSecs = [double]::NaN
    if ($line -match 'in ([\d.]+) s \(([\d.]+) iters/s\)') {
        $gpuSecs = [double]$Matches[1]; $gpuIps = [double]$Matches[2]
    }
    Write-Host ("  GPU: {0} iters in {1:N3}s  ->  {2:N1} iters/s" -f $hi, $gpuSecs, $gpuIps)

    $speedup = $gpuIps / $cpuIps
    Write-Host ("  speedup (GPU / CPU): {0:N1}x" -f $speedup) -ForegroundColor Green

    $rows += [pscustomobject]@{
        Spot        = $spot.name
        CPUThreads  = $threads
        "CPU it/s"  = [math]::Round($cpuIps, 1)
        "GPU it/s"  = [math]::Round($gpuIps, 1)
        Speedup     = ("{0:N1}x" -f $speedup)
    }
}

Write-Host "`n================ SUMMARY ================"
$rows | Format-Table -AutoSize | Out-String | Write-Host
Write-Host ("CPU = TexasSolverConsole (OpenMP, threads per row); GPU = river_gpu (RTX, CUDA).")
Write-Host ("Per-iteration throughput; isomorphism off on both. Speedup = GPU it/s / CPU it/s.")
