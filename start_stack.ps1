param(
    [ValidateSet("up", "down", "restart", "pull", "ps", "logs", "initdb")]
    [string]$Action = "up",
    [switch]$Build,
    [switch]$SkipInitTable,
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

function Invoke-InitDb {
    $tableName = if ($env:VECTOR_TABLE_NAME) { $env:VECTOR_TABLE_NAME } else { "embeddings" }
    $dimensions = if ($env:VECTOR_EMBEDDING_DIMENSIONS) { $env:VECTOR_EMBEDDING_DIMENSIONS } else { "768" }
    $sql = "CREATE EXTENSION IF NOT EXISTS vector; CREATE TABLE IF NOT EXISTS public.$tableName (id uuid PRIMARY KEY, metadata jsonb, contents text, embedding vector($dimensions));"
    Invoke-Compose -ComposeArgs @("exec", "-T", "timescaledb", "psql", "-U", "postgres", "-d", "postgres", "-c", $sql)
    Write-Host "Vector table 'public.$tableName' is ready."
}

switch ($Action) {
    "up" {
        $args = @("up", "-d")
        if ($Build) {
            $args += "--build"
        }
        Invoke-Compose -ComposeArgs $args
        if (-not $SkipInitTable) {
            Invoke-InitDb
        }
    }
    "down" {
        Invoke-Compose -ComposeArgs @("down")
    }
    "restart" {
        Invoke-Compose -ComposeArgs @("down")
        Invoke-Compose -ComposeArgs @("up", "-d")
        if (-not $SkipInitTable) {
            Invoke-InitDb
        }
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
    "initdb" {
        Invoke-InitDb
    }
}
