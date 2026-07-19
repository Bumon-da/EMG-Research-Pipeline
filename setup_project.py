from pathlib import Path

ROOT = Path.cwd()

folders = [
    "config",
    "data/raw",
    "data/processed",
    "data/external",

    "output/validation",
    "output/eda",
    "output/preprocessing",
    "output/features",
    "output/models",
    "output/reports",

    "src",
    "src/data",
    "src/eda",
    "src/preprocessing",
    "src/features",
    "src/models",
    "src/evaluation",
    "src/utils",

    "tests",
    "notebooks"
]

files = [
    "main.py",
    "README.md",
    ".gitignore",

    "config/__init__.py",
    "config/settings.py",
    "config/logging_config.py",

    "src/__init__.py",

    "src/data/__init__.py",
    "src/data/loader.py",
    "src/data/validator.py",
    "src/data/datamodels.py",

    "src/eda/__init__.py",
    "src/eda/analyzer.py",
    "src/eda/statistics.py",
    "src/eda/visualization.py",
    "src/eda/report.py",

    "src/preprocessing/__init__.py",

    "src/features/__init__.py",

    "src/models/__init__.py",

    "src/evaluation/__init__.py",

    "src/utils/__init__.py"
]

print("=" * 60)
print("Creating Project Structure")
print("=" * 60)

for folder in folders:
    path = ROOT / folder
    path.mkdir(parents=True, exist_ok=True)
    print(f"[DIR ] {folder}")

for file in files:
    path = ROOT / file

    if not path.exists():
        path.touch()
        print(f"[FILE] {file}")

print("\nProject structure created successfully!")