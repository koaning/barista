# Project Guidelines

## Running Python

Use `uv run` to execute Python scripts and marimo notebooks:

```bash
uv run marimo edit notebook.py
uv run python script.py
```

## Code Style

- Prefer `pathlib.Path` over `os.path` for file operations
- Use type hints
- Python 3.13+

## Marimo Notebooks

- Prefix cell-local variables with `_` to avoid conflicts across cells
- Let marimo's reactivity handle dependencies (don't guard with `if` statements)
- Final expression in a cell is what renders
- Preserve `column=N` in `@app.cell(column=N)` decorators - don't remove them
