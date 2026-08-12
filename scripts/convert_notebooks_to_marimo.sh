#!/usr/bin/env bash
# Converte notebooks Jupyter (.ipynb) em notebooks marimo (.py), preservando a
# separação de código por células.
#
# Uso:
#   scripts/convert_notebooks_to_marimo.sh                  # converte todos os .ipynb de notebooks/jupyter
#   scripts/convert_notebooks_to_marimo.sh caminho/para/x.ipynb [outro.ipynb ...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$REPO_ROOT/notebooks/jupyter"
DEST_DIR="$REPO_ROOT/notebooks/marimo"

if [ "$#" -gt 0 ]; then
    files=("$@")
else
    files=("$SRC_DIR"/*.ipynb)
fi

if [ "${#files[@]}" -eq 0 ] || [ ! -e "${files[0]}" ]; then
    echo "Nenhum arquivo .ipynb encontrado em: $SRC_DIR" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"

for ipynb in "${files[@]}"; do
    if [ ! -f "$ipynb" ]; then
        echo "Arquivo não encontrado, ignorando: $ipynb" >&2
        continue
    fi

    base="$(basename "$ipynb" .ipynb)"
    output="$DEST_DIR/$base.py"

    echo "Convertendo $ipynb -> $output"
    uv run --project "$REPO_ROOT" marimo -q -y convert "$ipynb" -o "$output"
done

echo "Conversão concluída."
