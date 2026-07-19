param(
    [ValidateSet("up", "down", "restart", "pull", "ps", "logs")]
    [string]$Action = "up",
    [switch]$Build,
    [switch]$Follow
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$composeFile = Join-Path $scriptRoot "docker\docker-compose.yml"

function Get-ComposeBaseCommand {
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        return @("docker", "compose")
    }

    if (Get-Command docker-compose -ErrorAction SilentlyContinue) {
        return @("docker-compose")
    }

    throw "Docker Compose was not found. Install Docker Desktop first."
}

function Invoke-Compose {
    param([string[]]$ComposeArgs)

    $base = Get-ComposeBaseCommand
    $full = @($base + @("-f", $composeFile) + $ComposeArgs)

    Write-Host "Running: $($full -join ' ')"
    & $full[0] $full[1..($full.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

switch ($Action) {
    "up" {
        $args = @("up", "-d")
        if ($Build) {
            $args += "--build"
        }
        Invoke-Compose -ComposeArgs $args
    }
    "down" {
        Invoke-Compose -ComposeArgs @("down")
    }
    "restart" {
        Invoke-Compose -ComposeArgs @("down")
        Invoke-Compose -ComposeArgs @("up", "-d")
    }
    "pull" {
        Invoke-Compose -ComposeArgs @("pull")
    }
    "ps" {
        Invoke-Compose -ComposeArgs @("ps")
    }
    "logs" {
        $args = @("logs")
        if ($Follow) {
            $args += "-f"
        }
        Invoke-Compose -ComposeArgs $args
    }
}
