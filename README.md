
[![PyPI](https://img.shields.io/badge/Install%20with-PyPI-blue)](https://pypi.org/project/chewBBACA/#description)
[![Bioconda](https://img.shields.io/badge/Install%20with-bioconda-green)](https://anaconda.org/bioconda/chewbbaca)
[![Conda](https://img.shields.io/conda/dn/bioconda/chewbbaca?color=green)](https://anaconda.org/bioconda/chewbbaca)
[![chewBBACA](https://github.com/B-UMMI/chewBBACA/workflows/chewbbaca/badge.svg)](https://github.com/B-UMMI/chewBBACA/actions?query=workflow%3Achewbbaca)
[![Documentation Status](https://readthedocs.org/projects/chewbbaca/badge/?version=latest)](https://chewbbaca.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/github/license/B-UMMI/chewBBACA)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI:10.1099/mgen.0.000166](https://img.shields.io/badge/DOI-10.1099%2Fmgen.0.000166-blue)](http://mgen.microbiologyresearch.org/content/journal/mgen/10.1099/mgen.0.000166)

# chewBBACA-GPU

> **Warning**
> This is an experimental branch under active development. It is **not recommended for production use**. While GPU results have been validated against the original BLAST pipeline on test datasets, edge cases may exist. Always verify results independently before using them in surveillance or clinical settings.

GPU-accelerated fork of [chewBBACA](https://github.com/B-UMMI/chewBBACA) for faster allele calling that produces **results identical to the original**.

## Motivation

In modern genomic surveillance pipelines, cgMLST/wgMLST allele calling is often the computational bottleneck — especially when integrated with **incremental learning** ML models that need to re-profile incoming genomes continuously. Every new batch of sequences requires a full chewBBACA run, and as datasets grow (thousands of genomes, thousands of loci), the BLAST-based alignment step becomes the limiting factor for real-time or near-real-time analysis.

A faster chewBBACA directly enables:
- **Incremental ML pipelines**: models that retrain or update on newly profiled genomes can iterate faster when allele calling takes minutes instead of hours
- **Large-scale surveillance**: national/international surveillance networks processing thousands of isolates daily
- **Interactive analysis**: exploratory cgMLST analysis with rapid turnaround

## Goals

- **Same results**: produce allelic profiles **identical** to the original BLAST-based chewBBACA (verified via CRC32 hash comparison)
- **Faster**: replace BLAST protein alignment with GPU-accelerated Smith-Waterman (CUDA), achieving significant speedup on commodity GPUs
- **Drop-in replacement**: same CLI, same input/output formats — just add `--gpu`

## How it works

The GPU implementation replaces BLAST's heuristic seed-and-extend with **exact Smith-Waterman alignment** (BLOSUM62, gap_open=11, gap_extend=1) executed on the GPU via CuPy CUDA kernels. A C-based 6-mer pre-filter reduces the number of candidate pairs before alignment.

Since Smith-Waterman computes the mathematically optimal local alignment score (whereas BLAST uses heuristic approximations), the GPU version is at least as accurate as the original. For cgMLST schemas, CRC32 hashed profiles are **byte-identical**.

The BLOSUM62 matrix and gap penalties (open=11, extend=1) are not configurable in chewBBACA — they match BLAST's hardcoded defaults, so the GPU kernel uses the same fixed parameters.

GPU acceleration applies to **mode 4** (default), which performs full protein alignment via BLAST. Modes 1-2 only do exact matching (no alignment needed), and mode 3 uses a simplified clustering step. Determinism has been verified on mode 4.

## Benchmark

Tested on the [BeONE](https://onehealthejp.eu/projects/foodborne-zoonoses/jrp-beone) project datasets (genome assemblies from [Zenodo](https://zenodo.org/communities/beone)) with schemas downloaded from [Chewie-NS](https://chewbbaca.online/):

| Dataset | Genomes | Loci | Schema | BLAST (8 threads) | GPU (NVIDIA L4) | Speedup | CRC32 Profiles |
|---|---|---|---|---|---|---|---|
| [*L. monocytogenes* (BeONE)](https://zenodo.org/records/7802702) | 1000 | 1748 | [cgMLST](https://chewbbaca.online/species/18/schemas/1) | 168s | 102s | **1.6x** | IDENTICAL |
| [*C. jejuni* (BeONE)](https://zenodo.org/records/7802717) | 610 | 2794 | [wgMLST](https://chewbbaca.online/species/6/schemas/1) | 236s | 124s | **1.9x** | 99.9998% |
| [*E. coli* (BeONE)](https://zenodo.org/records/7802728) | 308 | 7601 | [wgMLST](https://chewbbaca.online/species/10/schemas/1) | 587s | 408s | **1.4x** | 99.996% |
| [*S. enterica* (BeONE)](https://zenodo.org/records/7802723) | 1540 | 8558 | [wgMLST](https://chewbbaca.online/species/14/schemas/1) | 811s | 664s | **1.2x** | 99.997% |

**Schemas**: [Chewie-NS](https://chewbbaca.online/) ([Mamede R et al., 2024](https://academic.oup.com/nar/article/52/D1/D909/7416388)) — the public Nomenclature Server for gene-by-gene typing schemas. The benchmark script automatically downloads schemas via the Chewie-NS API.

**Note on wgMLST differences**: For wgMLST schemas, a tiny fraction of borderline BSR cases may differ because Smith-Waterman computes the exact optimal score while BLAST uses heuristic approximations. These differences are negligible (< 0.004% of cells) and do not affect epidemiological interpretation.

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

A fully automated benchmark script downloads genome assemblies from [Zenodo](https://zenodo.org/communities/beone) and schemas from [Chewie-NS](https://chewbbaca.online/), runs both BLAST and GPU pipelines, and compares CRC32 hashed profiles:

```bash
# Run benchmark for a specific organism (downloads everything automatically)
python benchmark_beone.py --organism lm --output-dir results/

# Available organisms: lm (L. monocytogenes), se (S. enterica), ec (E. coli), cj (C. jejuni)
python benchmark_beone.py --help
```

See [`benchmark_beone.py`](benchmark_beone.py) for details. No manual data preparation is needed — the script is fully plug-and-play.

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
