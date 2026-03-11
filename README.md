
[![PyPI](https://img.shields.io/badge/Install%20with-PyPI-blue)](https://pypi.org/project/chewBBACA/#description)
[![Bioconda](https://img.shields.io/badge/Install%20with-bioconda-green)](https://anaconda.org/bioconda/chewbbaca)
[![Conda](https://img.shields.io/conda/dn/bioconda/chewbbaca?color=green)](https://anaconda.org/bioconda/chewbbaca)
[![chewBBACA](https://github.com/B-UMMI/chewBBACA/workflows/chewbbaca/badge.svg)](https://github.com/B-UMMI/chewBBACA/actions?query=workflow%3Achewbbaca)
[![Documentation Status](https://readthedocs.org/projects/chewbbaca/badge/?version=latest)](https://chewbbaca.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/github/license/B-UMMI/chewBBACA)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI:10.1099/mgen.0.000166](https://img.shields.io/badge/DOI-10.1099%2Fmgen.0.000166-blue)](http://mgen.microbiologyresearch.org/content/journal/mgen/10.1099/mgen.0.000166)

# chewBBACA-GPU

GPU-accelerated fork of [chewBBACA](https://github.com/B-UMMI/chewBBACA) for faster allele calling with **deterministic results**.

## Goals

- **Determinism**: produce allelic profiles **identical** to the original BLAST-based pipeline (verified via CRC32 hash comparison)
- **Performance**: replace BLAST protein alignment with GPU-accelerated Smith-Waterman (CUDA), achieving significant speedup on commodity GPUs
- **Drop-in replacement**: same CLI, same input/output formats — just add `--gpu`

## How it works

The GPU implementation replaces BLAST's heuristic seed-and-extend with **exact Smith-Waterman alignment** (BLOSUM62, gap_open=11, gap_extend=1) executed on the GPU via CuPy CUDA kernels. A C-based 6-mer pre-filter reduces the number of candidate pairs before alignment.

Since Smith-Waterman computes the mathematically optimal local alignment score (whereas BLAST uses heuristic approximations), the GPU version is at least as accurate as the original. On all tested datasets, the allelic profiles are **byte-identical**.

## Benchmark

| Dataset | Genomes | Loci | BLAST (8 threads) | GPU (NVIDIA L4) | Speedup | Profiles |
|---|---|---|---|---|---|---|
| *L. monocytogenes* cgMLST (BeONE) | 1000 | 1748 | 168s | 108s | 1.56x | IDENTICAL |

## Quick start

### Requirements

- NVIDIA GPU with CUDA support
- Python >= 3.10
- CuPy (`pip install cupy-cuda12x` or appropriate version for your CUDA)
- GCC (to compile the C k-mer filter)

### Installation

```bash
# Clone this fork
git clone https://github.com/genpat-it/chewBBACA.git
cd chewBBACA
git checkout gpu-acceleration

# Install
pip install -e .

# Compile the C k-mer filter
gcc -O3 -march=native -shared -fPIC -o CHEWBBACA/utils/kmer_filter.so CHEWBBACA/utils/kmer_filter.c
```

### Usage

```bash
# Standard chewBBACA allele call with GPU acceleration
chewBBACA.py AlleleCall -i genomes/ -g schema/ -o output/ --gpu

# Without --gpu, behaves exactly like the original chewBBACA
chewBBACA.py AlleleCall -i genomes/ -g schema/ -o output/
```

## Reproducibility

A benchmark script is provided to verify determinism against the original BLAST pipeline using the [BeONE](https://onehealthejp.eu/projects/foodborne-zoonoses/jrp-beone) *Listeria monocytogenes* dataset:

```bash
python benchmark_beone.py --help
```

See [`benchmark_beone.py`](benchmark_beone.py) for details on downloading the dataset and running the comparison.

## Architecture

| File | Description |
|---|---|
| `CHEWBBACA/utils/gpu_sw.py` | CUDA Smith-Waterman kernel (CuPy RawKernel) |
| `CHEWBBACA/utils/blast_wrapper.py` | GPU/CPU dispatcher with C k-mer pre-filter |
| `CHEWBBACA/utils/core_functions.py` | GPU paths for `blast_clusters()` and self-score computation |
| `CHEWBBACA/utils/kmer_filter.c` | C extension for fast 6-mer candidate pair filtering |
| `CHEWBBACA/chewBBACA.py` | `--gpu` CLI flag |

## Original chewBBACA

**chewBBACA** is a software suite for the creation and evaluation of core genome and whole genome MultiLocus Sequence Typing (cg/wgMLST) schemas and results. The "BBACA" stands for "BSR-Based Allele Calling Algorithm". BSR stands for BLAST Score Ratio as proposed by [Rasko DA et al.](http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-6-2).

For full documentation of the original chewBBACA, see the [upstream repository](https://github.com/B-UMMI/chewBBACA) and [documentation](https://chewbbaca.readthedocs.io/en/latest/index.html).

## News

## 3.5.3 - 2026-03-10

- Fixed issue on the PrepExternalSchema module related to reading empty FASTA files after attempting to translate FASTA files from external schemas that contained no valid alleles. This issue did not affect the end result because the PrepExternalSchema module would detect that no alleles could be translated, skipping the next steps for that locus. However, not reading empty FASTA files avoids a warning raised by Biopython that could lead to errors in future releases.

- Add support for more recent versions of Numpy, SciPy, and Pandas (the versions of these dependencies were fixed to older versions due to past issues installing Pandas).

- Drop support for Python<=3.9. chewBBACA now requires Python>=3.10.

Check our [Changelog](https://github.com/B-UMMI/chewBBACA/blob/master/CHANGELOG.md) to learn about the latest changes.

## Citation

When using chewBBACA, please use the following citation:

> Silva M, Machado MP, Silva DN, Rossi M, Moran-Gilad J, Santos S, Ramirez M, Carriço JA. 2018. chewBBACA: A complete suite for gene-by-gene schema creation and strain identification. Microb Genom 4:000166. [doi:10.1099/mgen.0.000166](doi:10.1099/mgen.0.000166)
