import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = ROOT / "docker" / "docker-compose.yml"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage local agent stack (Ollama + TimescaleDB + pgAdmin4)."
    )
    parser.add_argument(
        "action",
        choices=["up", "down", "restart", "pull", "ps", "logs"],
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
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()

        if args.action == "up":
            compose_args = ["up", "-d"]
            if args.build:
                compose_args.append("--build")
            return run_compose(compose_args)

        if args.action == "down":
            return run_compose(["down"])

        if args.action == "restart":
            code = run_compose(["down"])
            if code != 0:
                return code
            return run_compose(["up", "-d"])

        if args.action == "pull":
            return run_compose(["pull"])

        if args.action == "ps":
            return run_compose(["ps"])

        if args.action == "logs":
            compose_args = ["logs"]
            if args.follow:
                compose_args.append("-f")
            return run_compose(compose_args)

        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
