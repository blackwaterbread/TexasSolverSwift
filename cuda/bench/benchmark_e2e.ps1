<#
.SYNOPSIS
  End-to-end wall-clock benchmark: total time a user waits for a strategy, CPU
  path vs GPU path, across iteration counts -> the data needed to define a hybrid
  (CPU-vs-GPU) router.

.DESCRIPTION
  Unlike benchmark.ps1 (pure per-iteration throughput), this measures the WHOLE
  pipeline each engine actually runs, including every fixed overhead:

    CPU path:  TexasSolverConsole  (compairer load + tree build + N CFR iters +
               EV pass + json dump)                          == cpu_total(N)

    GPU path:  SerializeRiver  (compairer load + tree build + write subgame)
             + river_gpu --dump (CUDA context init + load subgame + N CFR iters +
               dump + exploitability)                        == gpu_total(N)

  The GUI's "Solve on GPU" spawns both GPU processes fresh, so CUDA init and the
  serialize round-trip are paid every time and belong in gpu_total. For each spot
  we sweep N and report the break-even N where gpu_total(N) first beats cpu_total(N)
  (== where the GPU's per-iteration win amortizes its fixed overhead).

.PARAMETER Iters   Iteration counts to sweep (default 100 200 500 1000).
.PARAMETER Spots   Optional subset of spot names (default: all).
#>
param(
    [int[]]$Iters = @(100, 200, 500, 1000),
    [string[]]$Spots = @()
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repo
$env:PATH = "C:\Qt\5.15.2\mingw81_64\bin;C:\Qt\Tools\mingw810_64\bin;" + $env:PATH

$cpuExe = Join-Path $repo "cuda\cpu_export\build\release\TexasSolverConsole.exe"
$serExe = Join-Path $repo "cuda\cpu_export\build_ser\release\SerializeRiver.exe"
$gpuExe = Join-Path $repo "cuda\build\river_gpu.exe"
$resDir = Join-Path $repo "resources"
foreach ($e in @($cpuExe, $serExe, $gpuExe)) { if (-not (Test-Path $e)) { throw "missing executable: $e" } }

$allSpots = @(
    @{ name = "river";      cfg = "cuda\configs\river_spot.txt" },
    @{ name = "river_big";  cfg = "cuda\configs\river_big_spot.txt" },
    @{ name = "river_wide"; cfg = "cuda\configs\river_wide_spot.txt" },
    @{ name = "turn";       cfg = "cuda\configs\turn_spot.txt" },
    @{ name = "flop_mix";   cfg = "cuda\configs\flop_mix_spot.txt" }
)
if ($Spots.Count -gt 0) { $allSpots = $allSpots | Where-Object { $Spots -contains $_.name } }

function New-CpuConfig($srcRel, $iters, $dumpPath) {
    $src = Join-Path $repo $srcRel
    $out = New-Object System.Collections.Generic.List[string]
    foreach ($ln in (Get-Content $src)) {
        if     ($ln -match '^\s*set_max_iteration')  { $out.Add("set_max_iteration $iters") }
        elseif ($ln -match '^\s*set_print_interval') { $out.Add("set_print_interval 99999999") }
        elseif ($ln -match '^\s*set_accuracy')       { $out.Add("set_accuracy 0") }
        elseif ($ln -match '^\s*dump_result')        { $out.Add("dump_result $dumpPath") }
        else { $out.Add($ln) }
    }
    $tmp = Join-Path $env:TEMP ("e2e_cpu_{0}.txt" -f [IO.Path]::GetFileNameWithoutExtension($srcRel))
    Set-Content -Path $tmp -Value $out -Encoding ASCII
    return $tmp
}

# median of K wall-clock runs of a scriptblock, in seconds
function Measure-Med($k, $block) {
    $ts = @()
    for ($r = 0; $r -lt $k; $r++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $block | Out-Null
        $sw.Stop()
        $ts += $sw.Elapsed.TotalSeconds
    }
    return ($ts | Sort-Object)[[int]([math]::Floor($k / 2))]
}

$rows = @()
foreach ($spot in $allSpots) {
    Write-Host ("=== {0} ===" -f $spot.name) -ForegroundColor Cyan
    $subgame = Join-Path $env:TEMP ("e2e_{0}_sg.txt" -f $spot.name)
    $dump    = Join-Path $env:TEMP "e2e_dump.json"
    $cfgAbs  = Join-Path $repo $spot.cfg

    # SerializeRiver is N-independent: measure once (median of 2).
    $serT = Measure-Med 2 { & $serExe -i $cfgAbs -r $resDir -o $subgame *> $null }
    if (-not (Test-Path $subgame)) { Write-Host "  [skip] serialize produced no subgame"; continue }
    Write-Host ("  SerializeRiver (fixed): {0:N3}s" -f $serT)

    $breakeven = $null
    foreach ($n in $Iters) {
        $cpuCfg = New-CpuConfig $spot.cfg $n $dump
        $cpuT = Measure-Med 2 { & $cpuExe -i $cpuCfg -r $resDir *> $null }
        $gpuSolveT = Measure-Med 2 { & $gpuExe -s $subgame -d $dump -n $n *> $null }
        $gpuT = $serT + $gpuSolveT
        $win = if ($gpuT -lt $cpuT) { "GPU" } else { "CPU" }
        if (-not $breakeven -and $gpuT -lt $cpuT) { $breakeven = $n }
        Write-Host ("  n={0,5}:  CPU {1,7:N3}s   GPU {2,7:N3}s (ser {3:N2}+solve {4:N2})   -> {5}" -f `
                    $n, $cpuT, $gpuT, $serT, $gpuSolveT, $win)
        $rows += [pscustomobject]@{
            Spot = $spot.name; N = $n
            "CPU s" = [math]::Round($cpuT, 3)
            "GPU s" = [math]::Round($gpuT, 3)
            "GPU ser+solve" = ("{0:N2}+{1:N2}" -f $serT, $gpuSolveT)
            Faster = $win
        }
    }
    $beTxt = if ($breakeven) { "GPU wins from n>=$breakeven" } else { "CPU wins across all tested n" }
    Write-Host ("  break-even: {0}" -f $beTxt) -ForegroundColor Green
}

Write-Host "`n================ END-TO-END SUMMARY ================"
$rows | Format-Table -AutoSize | Out-String | Write-Host
Write-Host "CPU s = TexasSolverConsole total; GPU s = SerializeRiver + river_gpu --dump total (incl. CUDA init + exploitability)."
Write-Host "Whole-pipeline wall-clock a user waits for. Use the break-even per spot to set the hybrid router threshold."
