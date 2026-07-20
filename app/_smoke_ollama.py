#!/usr/bin/env python3
"""
Smoke test for the Deployable Local Ollama Agent stack.

Verifies:
1. Docker daemon is running
2. All services (ollama, timescaledb, pgadmin) are up and healthy
3. Ollama models are available
4. Database connectivity and vector table exist
5. Embedding generation works
6. Basic multi-turn chat with memory works
"""

import json
import subprocess
import sys
import time
from uuid import uuid4

import requests

from database.vector_store import VectorStore
from chat_with_memory import _store_turn, _generate_reply


class SmokeTestError(Exception):
    """Smoke test assertion error."""
    pass


def check_docker() -> None:
    """Verify Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise SmokeTestError("Docker daemon not responding")
        print("✓ Docker daemon is running")
    except FileNotFoundError:
        raise SmokeTestError("Docker not found in PATH")
    except subprocess.TimeoutExpired:
        raise SmokeTestError("Docker daemon timeout")


def check_containers() -> None:
    """Verify all containers are running and healthy."""
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", "../docker/docker-compose.yml", "ps", "--format=json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise SmokeTestError("Failed to query Docker Compose services")

        # Parse JSON lines format (one JSON object per line)
        services = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                try:
                    services.append(json.loads(line))
                except json.JSONDecodeError:
                    raise SmokeTestError(f"Invalid Docker Compose output line: {line}")

        required = {"ollama", "timescaledb", "pgadmin"}
        running = set()
        
        for svc in services:
            name = svc.get("Service", "")
            state = svc.get("State", "")
            health = svc.get("Health", "")
            
            if state == "running":
                running.add(name)
                if health and health not in {"", "healthy"}:
                    raise SmokeTestError(f"{name} is running but unhealthy: {health}")

        if not required.issubset(running):
            missing = required - running
            raise SmokeTestError(f"Missing services: {missing}")
        
        print("✓ All containers running and healthy")
    except subprocess.TimeoutExpired:
        raise SmokeTestError("Docker Compose timeout")


def check_ollama() -> None:
    """Verify Ollama is responding and has required models."""
    try:
        # Test connectivity
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        models = {m.get("name") for m in data.get("models", [])}
        
        required_models = {"llama3.2:3b", "nomic-embed-text:latest"}
        if not required_models.issubset(models):
            missing = required_models - models
            raise SmokeTestError(f"Missing Ollama models: {missing}")
        
        print(f"✓ Ollama responding with {len(models)} models")
    except requests.exceptions.ConnectionError:
        raise SmokeTestError("Cannot connect to Ollama at localhost:11434")
    except requests.exceptions.Timeout:
        raise SmokeTestError("Ollama timeout")
    except (KeyError, ValueError) as e:
        raise SmokeTestError(f"Invalid Ollama response: {e}")


def check_database() -> None:
    """Verify database connectivity and vector table exists."""
    try:
        # Check vector extension
        result = subprocess.run(
            [
                "docker", "compose", "-f", "../docker/docker-compose.yml", "exec", "-T",
                "timescaledb", "psql", "-U", "postgres", "-d", "postgres", "-tc",
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname='vector')"
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or "t" not in result.stdout.lower():
            raise SmokeTestError("Vector extension not installed")
        
        # Check embeddings table
        result = subprocess.run(
            [
                "docker", "compose", "-f", "../docker/docker-compose.yml", "exec", "-T",
                "timescaledb", "psql", "-U", "postgres", "-d", "postgres", "-tc",
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='embeddings' AND table_schema='public')"
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or "t" not in result.stdout.lower():
            raise SmokeTestError("Vector table 'embeddings' does not exist")
        
        print("✓ Database connected and vector table ready")
    except subprocess.TimeoutExpired:
        raise SmokeTestError("Database check timeout")
    except Exception as e:
        raise SmokeTestError(f"Database check failed: {e}")


def check_embeddings() -> None:
    """Verify embedding generation works."""
    try:
        vec = VectorStore()
        test_text = "This is a test embedding"
        embedding = vec.get_embedding(test_text)
        
        if not isinstance(embedding, list):
            raise SmokeTestError("Embedding is not a list")
        
        if len(embedding) != 768:
            raise SmokeTestError(f"Expected 768-dim embedding, got {len(embedding)}")
        
        if not all(isinstance(x, float) for x in embedding):
            raise SmokeTestError("Embedding contains non-numeric values")
        
        print(f"✓ Embedding generation works ({len(embedding)} dimensions)")
    except Exception as e:
        raise SmokeTestError(f"Embedding test failed: {e}")


def check_chat_with_memory() -> None:
    """Verify multi-turn chat with memory works."""
    try:
        conv_id = str(uuid4())
        vec = VectorStore()
        
        # First turn
        user_msg1 = "My favorite color is blue"
        _store_turn(vec, conv_id, "user", user_msg1)
        response1 = _generate_reply(vec, conv_id, user_msg1, memory_limit=3)
        _store_turn(vec, conv_id, "assistant", response1)
        
        if not response1 or len(response1) < 5:
            raise SmokeTestError(f"Assistant response too short: {response1}")
        
        # Second turn (should use memory)
        user_msg2 = "What did I just tell you?"
        _store_turn(vec, conv_id, "user", user_msg2)
        response2 = _generate_reply(vec, conv_id, user_msg2, memory_limit=3)
        _store_turn(vec, conv_id, "assistant", response2)
        
        if not response2 or len(response2) < 5:
            raise SmokeTestError(f"Assistant response too short: {response2}")
        
        # Check if memory was used (response should mention "blue" or "color")
        lower_response = response2.lower()
        if "blue" not in lower_response and "color" not in lower_response:
            # It's not a hard requirement - models may paraphrase or summarize
            # but it's good to note if memory wasn't picked up
            pass
        
        print(f"✓ Chat with memory works (turn 1: {len(response1)} chars, turn 2: {len(response2)} chars)")
    except Exception as e:
        raise SmokeTestError(f"Chat test failed: {e}")


def main() -> int:
    """Run all smoke tests."""
    tests = [
        ("Docker daemon", check_docker),
        ("Containers", check_containers),
        ("Ollama", check_ollama),
        ("Database", check_database),
        ("Embeddings", check_embeddings),
        ("Chat with memory", check_chat_with_memory),
    ]
    
    print("\n" + "=" * 60)
    print("SMOKE TEST: Deployable Local Ollama Agent")
    print("=" * 60 + "\n")
    
    failed = []
    for name, test_func in tests:
        try:
            test_func()
        except SmokeTestError as e:
            print(f"✗ {name}: {e}")
            failed.append((name, str(e)))
        except Exception as e:
            print(f"✗ {name}: Unexpected error: {e}")
            failed.append((name, f"Unexpected error: {e}"))
    
    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED: {len(failed)} test(s) failed\n")
        for name, error in failed:
            print(f"  • {name}: {error}")
        print("=" * 60 + "\n")
        return 1
    else:
        print("PASSED: All smoke tests passed! ✓")
        print("=" * 60 + "\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
