"""
GitHub repository ingestion tool.

Clones a GitHub repository and ingests code files into the memory database.
Each file becomes a memory cell, allowing the AI to recall code context.
"""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from tools import BaseTool


class GitHubIngestTool(BaseTool):
    """Tool to ingest GitHub repositories into memory."""

    name = "ingest"
    description = "Clone and ingest a GitHub repository into memory"
    command = "/ingest"

    CODE_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".cs",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".html",
        ".css",
        ".scss",
        ".sass",
        ".vue",
        ".svelte",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".md",
        ".txt",
        ".sql",
        ".sh",
        ".bash",
        ".zsh",
        ".dockerfile",
        ".makefile",
    }

    SKIP_DIRS = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "build",
        "dist",
        "target",
        ".idea",
        ".vscode",
        ".cache",
        "vendor",
        "packages",
        ".next",
        ".nuxt",
    }

    MAX_FILE_SIZE = 500 * 1024

    def execute(self, url: str = None, **kwargs) -> Dict[str, Any]:
        if not url:
            return {"success": False, "error": "No URL provided"}

        if not url.startswith(("https://github.com/", "git@github.com:")):
            return {"success": False, "error": "Invalid GitHub URL"}

        print(f"\nIngesting repository: {url}")
        temp_dir = tempfile.mkdtemp(prefix="neurosavant_ingest_")

        try:
            print("   Cloning repository...")
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, temp_dir],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                return {"success": False, "error": f"Clone failed: {result.stderr}"}

            repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
            root_cell_id = f"repo_root_{repo_name}"
            self._create_repo_root(root_cell_id, repo_name, url)
            print(f"   Created repository root: {root_cell_id}")

            print("   Scanning for files...")
            all_files = self._find_code_files(temp_dir)
            total_files = len(all_files)
            print(f"   Found {total_files} files to ingest")

            if total_files == 0:
                return {
                    "success": True,
                    "files_ingested": 0,
                    "total_characters": 0,
                    "repository": repo_name,
                    "message": "No code files found",
                }

            files_ingested = 0
            total_chars = 0
            start_time = time.time()
            batch_size = 50
            file_batch = []

            for index, filepath in enumerate(all_files):
                try:
                    content = self._read_file(filepath)
                    if content:
                        rel_path = os.path.relpath(filepath, temp_dir)
                        cell_id = f"repo_{repo_name}_{rel_path.replace('/', '_')}"[:50]
                        file_batch.append((cell_id, rel_path, content))
                        files_ingested += 1
                        total_chars += len(content)
                except Exception:
                    continue

                pct = (index + 1) / total_files * 100
                bar_width = 30
                filled = int(bar_width * (index + 1) // total_files)
                bar = "#" * filled + "." * (bar_width - filled)
                elapsed = time.time() - start_time
                rate = (index + 1) / elapsed if elapsed > 0 else 0
                eta = (total_files - index - 1) / rate if rate > 0 else 0
                print(
                    f"\r   [{bar}] {index + 1}/{total_files} ({pct:.0f}%) | {rate:.1f} files/s | ETA: {eta:.0f}s",
                    end="",
                    flush=True,
                )

                if len(file_batch) >= batch_size:
                    for cell_id, rel_path, content in file_batch:
                        self._store_file(cell_id, rel_path, content, repo_name, parent_id=root_cell_id)
                    file_batch = []

            for cell_id, rel_path, content in file_batch:
                self._store_file(cell_id, rel_path, content, repo_name, parent_id=root_cell_id)

            print()
            elapsed = time.time() - start_time
            print(f"   Ingested {files_ingested} files ({total_chars:,} chars) in {elapsed:.1f}s")

            return {
                "success": True,
                "files_ingested": files_ingested,
                "total_characters": total_chars,
                "repository": repo_name,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Clone timed out (>120s)"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _find_code_files(self, root_dir: str) -> List[str]:
        code_files = []

        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [directory for directory in dirs if directory.lower() not in self.SKIP_DIRS]

            for filename in files:
                ext = Path(filename).suffix.lower()
                if ext in self.CODE_EXTENSIONS or filename.lower() in ("makefile", "dockerfile", "readme"):
                    filepath = os.path.join(root, filename)
                    try:
                        if os.path.getsize(filepath) <= self.MAX_FILE_SIZE:
                            code_files.append(filepath)
                    except OSError:
                        pass

        return code_files

    def _read_file(self, filepath: str) -> str:
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(filepath, "r", encoding=encoding) as handle:
                    return handle.read()
            except Exception:
                continue

        return None

    def _store_file(self, cell_id: str, filepath: str, content: str, repo_name: str, parent_id: str = None):
        if not self.memory_grid:
            return

        header = f"[Repository: {repo_name}]\n[File: {filepath}]\n\n"
        full_content = header + content[:4000]

        try:
            self.memory_grid.ingest(full_content)
        except Exception:
            pass

    def _create_repo_root(self, root_cell_id: str, repo_name: str, url: str):
        if not self.memory_grid:
            return

        root_content = f"""[Repository Root: {repo_name}]
[Source: {url}]
[Type: GitHub Repository]

This is a summary node for the {repo_name} repository.
All files from this repository are connected as children of this node.
Use this node to navigate the repository structure hierarchically.
"""

        try:
            self.memory_grid.ingest(root_content)
        except Exception as exc:
            print(f"   Failed to create repo root: {exc}")

    def help(self) -> str:
        return """GitHub Ingestion Tool

Usage: /ingest <github-url>

Examples:
  /ingest https://github.com/user/repo
  /ingest https://github.com/microsoft/vscode

Supported file types: Python, JavaScript, TypeScript, Java, C/C++, Go, Rust, etc.
Skips: node_modules, .git, __pycache__, build folders, files >500KB
"""
