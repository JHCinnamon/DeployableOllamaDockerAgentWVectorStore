import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = ROOT / "docker" / "docker-compose.yml"
DEFAULT_TABLE_NAME = os.getenv("VECTOR_TABLE_NAME", "embeddings")
DEFAULT_EMBEDDING_DIMENSIONS = int(os.getenv("VECTOR_EMBEDDING_DIMENSIONS", "1536"))


def docker_compose_command() -> list[str]:
    """Return docker compose base command using the installed CLI variant."""
    if shutil.which("docker"):
        return ["docker", "compose"]
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    raise RuntimeError("Docker Compose was not found. Install Docker Desktop first.")


def run_compose(args: list[str]) -> int:
    base = docker_compose_command()
    command = [*base, "-f", str(COMPOSE_FILE), *args]
    print("Running:", " ".join(command))
    result = subprocess.run(command, cwd=ROOT)
    return result.returncode


def run_initdb() -> int:
    base = docker_compose_command()
    sql = (
        "CREATE EXTENSION IF NOT EXISTS vector;"
        f"CREATE TABLE IF NOT EXISTS public.{DEFAULT_TABLE_NAME} ("
        "id uuid PRIMARY KEY, "
        "metadata jsonb, "
        "contents text, "
        f"embedding vector({DEFAULT_EMBEDDING_DIMENSIONS})"
        ");"
    )
    command = [
        *base,
        "-f",
        str(COMPOSE_FILE),
        "exec",
        "-T",
        "timescaledb",
        "psql",
        "-U",
        "postgres",
        "-d",
        "postgres",
        "-c",
        sql,
    ]
    print("Running:", " ".join(command))
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode == 0:
        print(f"Vector table 'public.{DEFAULT_TABLE_NAME}' is ready.")
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage local agent stack (Ollama + TimescaleDB + pgAdmin4)."
    )
    parser.add_argument(
        "action",
        choices=["up", "down", "restart", "pull", "ps", "logs", "initdb"],
        help="Compose action to run",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Use --build when running action=up",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Use -f when running action=logs",
    )
    parser.add_argument(
        "--skip-init-table",
        action="store_true",
        help="Skip vector table initialization after action=up or action=restart",
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()

        if args.action == "up":
            compose_args = ["up", "-d"]
            if args.build:
                compose_args.append("--build")
            code = run_compose(compose_args)
            if code != 0:
                return code
            if not args.skip_init_table:
                return run_initdb()
            return 0

        if args.action == "down":
            return run_compose(["down"])

        if args.action == "restart":
            code = run_compose(["down"])
            if code != 0:
                return code
            code = run_compose(["up", "-d"])
            if code != 0:
                return code
            if not args.skip_init_table:
                return run_initdb()
            return 0

        if args.action == "pull":
            return run_compose(["pull"])

        if args.action == "ps":
            return run_compose(["ps"])

        if args.action == "logs":
            compose_args = ["logs"]
            if args.follow:
                compose_args.append("-f")
            return run_compose(compose_args)

        if args.action == "initdb":
            return run_initdb()

        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
