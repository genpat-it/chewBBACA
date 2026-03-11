#!/bin/bash
# Run all BeONE benchmarks using local data.
# Downloads schemas from Chewie-NS if not present.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate chewbbacca_gpu

# Compile C k-mer filter if needed
if [ ! -f CHEWBBACA/utils/kmer_filter.so ]; then
    echo "Compiling kmer_filter.so..."
    gcc -O3 -march=native -shared -fPIC -o CHEWBBACA/utils/kmer_filter.so CHEWBBACA/utils/kmer_filter.c
fi

OUTPUT_BASE="/mnt/disk2/a.deruvo/beone_benchmarks"
mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "Running Listeria monocytogenes benchmark"
echo "========================================"
python benchmark_beone.py \
    --organism lm \
    --data-dir "$OUTPUT_BASE/data" \
    --output-dir "$OUTPUT_BASE/output" \
    --cpu-cores 8 \
    --skip-download

echo ""
echo "========================================"
echo "Running Salmonella enterica benchmark"
echo "========================================"
python benchmark_beone.py \
    --organism se \
    --data-dir "$OUTPUT_BASE/data" \
    --output-dir "$OUTPUT_BASE/output" \
    --cpu-cores 8 \
    --skip-download

echo ""
echo "========================================"
echo "Running Escherichia coli benchmark"
echo "========================================"
python benchmark_beone.py \
    --organism ec \
    --data-dir "$OUTPUT_BASE/data" \
    --output-dir "$OUTPUT_BASE/output" \
    --cpu-cores 8

echo ""
echo "========================================"
echo "Running Campylobacter jejuni benchmark"
echo "========================================"
python benchmark_beone.py \
    --organism cj \
    --data-dir "$OUTPUT_BASE/data" \
    --output-dir "$OUTPUT_BASE/output" \
    --cpu-cores 8
