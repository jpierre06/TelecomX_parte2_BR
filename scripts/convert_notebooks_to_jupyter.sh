#!/usr/bin/env bash
# Converte notebooks marimo (.py) em notebooks Jupyter (.ipynb), preservando a
# separação de código por células (ordem top-down, sem executar o notebook).
#
# Requer o pacote nbformat (dependência dev do projeto: `uv add --group dev nbformat`).
#
# Uso:
#   scripts/convert_notebooks_to_jupyter.sh                 # converte todos os .py de notebooks/marimo
#   scripts/convert_notebooks_to_jupyter.sh caminho/para/x.py [outro.py ...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$REPO_ROOT/notebooks/marimo"
DEST_DIR="$REPO_ROOT/notebooks/jupyter"

if [ "$#" -gt 0 ]; then
    files=("$@")
else
    files=("$SRC_DIR"/*.py)
fi

if [ "${#files[@]}" -eq 0 ] || [ ! -e "${files[0]}" ]; then
    echo "Nenhum arquivo .py encontrado em: $SRC_DIR" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"

for py in "${files[@]}"; do
    if [ ! -f "$py" ]; then
        echo "Arquivo não encontrado, ignorando: $py" >&2
        continue
    fi

    base="$(basename "$py" .py)"
    output="$DEST_DIR/$base.ipynb"

    echo "Convertendo $py -> $output"
    uv run --project "$REPO_ROOT" marimo export ipynb "$py" --sort top-down -o "$output" -f
done

echo "Conversão concluída."
