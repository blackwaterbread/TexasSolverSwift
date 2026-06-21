<#
.SYNOPSIS
  Validate the --engine auto router: does it actually pick the faster engine?

.DESCRIPTION
  Drives the unified CLI (TexasSolverConsole -e cpu|gpu|auto) end-to-end for each
  spot and times the WHOLE user-facing wall-clock of each path:
    cpu  : in-process CPU CFR solve + dump
    gpu  : spawn SerializeRiver + river_gpu --dump, load back, dump (incl. CUDA init)
    auto : the router chooses one of the above by board size (+ turn iter threshold)

  For every spot we report cpu/gpu/auto totals, which engine auto resolved to, and
  whether that choice is the empirically faster one (PASS = auto picked min(cpu,gpu)).
  A turn sub-sweep shows the router flipping cpu<->gpu around its iteration threshold.

  All runs force set_dump_rounds 1 (root street) so every engine dumps the same
  thing and the GPU path is fully correct (no multi-street placeholder caveat), and
  redirect dump_result to TEMP so the golden files are never touched.

.PARAMETER Spots   Optional subset of spot names (default: all).
.PARAMETER Repeat  Timed runs per cell; reports the median (default 2).
#>
param(
    [string[]]$Spots = @(),
    [int]$Repeat = 2
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repo
$env:PATH = "C:\Qt\5.15.2\mingw81_64\bin;C:\Qt\Tools\mingw810_64\bin;" + $env:PATH

$con = Join-Path $repo "cuda\cpu_export\build\release\TexasSolverConsole.exe"
if (-not (Test-Path $con)) { throw "missing console exe: $con" }

# board size -> engine auto WILL pick (mirrors CommandLineTool::resolveEngine)
function Expect-Engine($boardCards, $iters) {
    if ($boardCards -eq 5) { return "cpu" }            # river
    if ($boardCards -eq 4) { if ($iters -ge 100) { return "gpu" } else { return "cpu" } }  # turn
    return "gpu"                                       # flop
}

# name -> (cpu config, board cards, iteration count for the run)
$allSpots = @(
    @{ name = "river";      cfg = "cuda\configs\river_spot.txt";      board = 5; iters = 200 },
    @{ name = "river_big";  cfg = "cuda\configs\river_big_spot.txt";  board = 5; iters = 200 },
    @{ name = "river_wide"; cfg = "cuda\configs\river_wide_spot.txt"; board = 5; iters = 200 },
    @{ name = "turn";       cfg = "cuda\configs\turn_spot.txt";       board = 4; iters = 500 },
    @{ name = "flop_mix";   cfg = "cuda\configs\flop_mix_spot.txt";   board = 3; iters = 300 }
)
if ($Spots.Count -gt 0) { $allSpots = $allSpots | Where-Object { $Spots -contains $_.name } }

# temp config: override iterations, force dump_rounds 1, redirect dump to TEMP
function Make-Config($srcRel, $iters) {
    $dump = Join-Path $env:TEMP "auto_dump.json"
    $out = New-Object System.Collections.Generic.List[string]
    $hasDr = $false
    foreach ($ln in (Get-Content (Join-Path $repo $srcRel))) {
        if     ($ln -match '^\s*set_max_iteration')  { $out.Add("set_max_iteration $iters") }
        elseif ($ln -match '^\s*set_dump_rounds')    { $out.Add("set_dump_rounds 1"); $hasDr = $true }
        elseif ($ln -match '^\s*dump_result')        { $out.Add("dump_result $dump") }
        else { $out.Add($ln) }
    }
    if (-not $hasDr) { $out.Insert($out.Count - 1, "set_dump_rounds 1") }
    $tmp = Join-Path $env:TEMP ("auto_cfg_{0}.txt" -f [IO.Path]::GetFileNameWithoutExtension($srcRel))
    Set-Content -Path $tmp -Value $out -Encoding ASCII
    return $tmp
}

# run the CLI once per repeat; return median seconds + the engine auto/forced resolved to
function Run-CLI($cfg, $engine, $repeat) {
    $ts = @(); $resolved = $null
    for ($r = 0; $r -lt $repeat; $r++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $out = & $con -i $cfg -r $resDirArg -e $engine 2>&1
        $sw.Stop()
        $ts += $sw.Elapsed.TotalSeconds
        if (-not $resolved) {
            $m = $out | Select-String 'engine=(\w+)'
            if ($m) { $resolved = $m[0].Matches[0].Groups[1].Value }
        }
    }
    return @{ secs = ($ts | Sort-Object)[[int]([math]::Floor($repeat / 2))]; engine = $resolved }
}
$resDirArg = Join-Path $repo "resources"

$rows = @()
foreach ($spot in $allSpots) {
    Write-Host ("=== {0} (board {1}, n={2}) ===" -f $spot.name, $spot.board, $spot.iters) -ForegroundColor Cyan
    $cfg = Make-Config $spot.cfg $spot.iters
    $cpu  = Run-CLI $cfg "cpu"  $Repeat
    $gpu  = Run-CLI $cfg "gpu"  $Repeat
    $auto = Run-CLI $cfg "auto" $Repeat

    $fasterEng = if ($cpu.secs -le $gpu.secs) { "cpu" } else { "gpu" }
    $expect    = Expect-Engine $spot.board $spot.iters
    $rulePass  = if ($auto.engine -eq $expect) { "ok" } else { "MISMATCH(exp $expect)" }
    $pickPass  = if ($auto.engine -eq $fasterEng) { "PASS" } else { "SLOWER" }

    Write-Host ("  cpu {0,7:N3}s   gpu {1,7:N3}s   auto->{2} {3,7:N3}s   faster={4}  rule={5}  pick={6}" -f `
                $cpu.secs, $gpu.secs, $auto.engine, $auto.secs, $fasterEng, $rulePass, $pickPass) `
                -ForegroundColor $(if ($pickPass -eq "PASS") { "Green" } else { "Yellow" })

    $rows += [pscustomobject]@{
        Spot = $spot.name; Board = $spot.board; N = $spot.iters
        "CPU s" = [math]::Round($cpu.secs, 3)
        "GPU s" = [math]::Round($gpu.secs, 3)
        "Auto" = $auto.engine
        "Auto s" = [math]::Round($auto.secs, 3)
        Faster = $fasterEng
        Pick = $pickPass
    }
}

# ---- turn iteration-threshold sweep: auto should flip cpu->gpu around n=100 ----
$turn = $allSpots | Where-Object { $_.name -eq "turn" }
if ($turn) {
    Write-Host "`n=== turn threshold sweep (auto flips at max_iteration >= 100) ===" -ForegroundColor Cyan
    foreach ($n in @(50, 90, 120, 300)) {
        $cfg = Make-Config $turn.cfg $n
        $a = Run-CLI $cfg "auto" 1
        $exp = Expect-Engine 4 $n
        $mark = if ($a.engine -eq $exp) { "ok" } else { "MISMATCH(exp $exp)" }
        Write-Host ("  n={0,4}: auto -> {1}  [{2}]" -f $n, $a.engine, $mark)
    }
}

Write-Host "`n================ AUTO ROUTER SUMMARY ================"
$rows | Format-Table -AutoSize | Out-String | Write-Host
Write-Host "Whole-CLI wall-clock (cpu/gpu = forced; auto = router's choice). Pick=PASS means auto chose the faster engine."
Write-Host "dump_rounds forced to 1 (root street) so all engines dump identically and the GPU path is fully correct."
