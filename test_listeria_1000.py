#!/usr/bin/env python3
"""Benchmark: GPU vs BLAST on 1000 Listeria genomes with cgMLST schema."""

import os
import sys
import time
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

# Paths
BASE = os.path.dirname(__file__)
SCHEMA = os.path.join(BASE, 'listeria_schema', 'Listeria_monocytogenes_cgMLST')
GENOMES_DIR = '/mnt/backup/cgDist-datasets/Lm_1462'
N_SAMPLES = 1000
CPU_CORES = 8


def run_allele_call(output_dir, schema_dir, use_gpu=False):
    """Run chewBBACA allele calling."""
    from CHEWBBACA.utils import (constants as ct,
                                  blast_wrapper as bw,
                                  file_operations as fo)
    from CHEWBBACA.AlleleCall import allele_call

    bw.disable_gpu()
    if use_gpu:
        bw.enable_gpu()

    # Build genome list (first N_SAMPLES)
    genome_list = os.path.join(output_dir, 'genomes.txt')
    all_fastas = sorted([f for f in os.listdir(GENOMES_DIR) if f.endswith('.fasta')])
    genome_files = [os.path.join(GENOMES_DIR, f) for f in all_fastas[:N_SAMPLES]]
    with open(genome_list, 'w') as f:
        f.write('\n'.join(genome_files))
    print(f"  Genomes: {len(genome_files)}")

    # Build loci list
    loci_list = os.path.join(output_dir, 'loci.txt')
    schema_files = [os.path.join(schema_dir, f)
                    for f in os.listdir(schema_dir)
                    if f.endswith('.fasta')]
    with open(loci_list, 'w') as f:
        f.write('\n'.join(schema_files))
    print(f"  Loci: {len(schema_files)}")

    # Load schema config
    config_file = os.path.join(schema_dir, ct.SCHEMA_CONFIG_BASENAME)
    schema_params = fo.pickle_loader(config_file)

    def unwrap(val, default):
        if isinstance(val, list):
            return val[0] if val else default
        return val if val is not None else default

    config = {
        'Minimum sequence length': unwrap(schema_params.get('minimum_locus_length'), ct.MINIMUM_LENGTH_DEFAULT),
        'Size threshold': unwrap(schema_params.get('size_threshold'), ct.SIZE_THRESHOLD_DEFAULT),
        'Translation table': unwrap(schema_params.get('translation_table'), ct.GENETIC_CODES_DEFAULT),
        'BLAST Score Ratio': unwrap(schema_params.get('bsr'), ct.DEFAULT_BSR),
        'Word size': ct.WORD_SIZE_DEFAULT,
        'Window size': ct.WINDOW_SIZE_DEFAULT,
        'Clustering similarity': ct.CLUSTERING_SIMILARITY_DEFAULT,
        'Prodigal training file': None,
        'CPU cores': CPU_CORES,
        'BLAST path': '',
        'CDS input': False,
        'Prodigal mode': 'single',
        'Mode': 4,
    }

    ptf_files = [os.path.join(schema_dir, f) for f in os.listdir(schema_dir) if f.endswith('.trn')]
    if ptf_files:
        config['Prodigal training file'] = ptf_files[0]

    allele_call.main(genome_list, loci_list, schema_dir,
                     output_dir, False, False, False,
                     False, False, False, 'crc32', False, config)
    return output_dir


def read_file_lines(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [line.strip() for line in f]


def find_hashed_file(output_dir):
    for f in os.listdir(output_dir):
        if 'hashed' in f.lower() and f.endswith('.tsv'):
            return os.path.join(output_dir, f)
    return None


def main():
    print("=" * 70)
    print(f"LISTERIA BENCHMARK: {N_SAMPLES} samples, cgMLST schema (1748 loci), {CPU_CORES} threads")
    print("=" * 70)

    # Independent schema copies
    blast_schema = tempfile.mkdtemp(prefix=f'schema_blast_{N_SAMPLES}_', dir='/mnt/disk2/a.deruvo')
    gpu_schema = tempfile.mkdtemp(prefix=f'schema_gpu_{N_SAMPLES}_', dir='/mnt/disk2/a.deruvo')
    print(f"\nCopying schema...")
    shutil.copytree(SCHEMA, os.path.join(blast_schema, 'schema'), dirs_exist_ok=True)
    shutil.copytree(SCHEMA, os.path.join(gpu_schema, 'schema'), dirs_exist_ok=True)
    blast_schema_dir = os.path.join(blast_schema, 'schema')
    gpu_schema_dir = os.path.join(gpu_schema, 'schema')
    print(f"  Done.")

    # BLAST
    print("\n" + "-" * 70)
    print("Running BLAST version...")
    print("-" * 70)
    blast_dir = tempfile.mkdtemp(prefix=f'chewie_{N_SAMPLES}_blast_', dir='/mnt/disk2/a.deruvo')
    t0 = time.time()
    try:
        run_allele_call(blast_dir, blast_schema_dir, use_gpu=False)
    except Exception as e:
        print(f"  BLAST run failed: {e}")
        import traceback; traceback.print_exc()
    blast_time = time.time() - t0
    print(f"  BLAST time: {blast_time:.1f}s")

    # GPU
    print("\n" + "-" * 70)
    print("Running GPU version...")
    print("-" * 70)
    gpu_dir = tempfile.mkdtemp(prefix=f'chewie_{N_SAMPLES}_gpu_', dir='/mnt/disk2/a.deruvo')
    t0 = time.time()
    try:
        run_allele_call(gpu_dir, gpu_schema_dir, use_gpu=True)
    except Exception as e:
        print(f"  GPU run failed: {e}")
        import traceback; traceback.print_exc()
    gpu_time = time.time() - t0
    print(f"  GPU time: {gpu_time:.1f}s")

    # Cleanup schema copies
    shutil.rmtree(blast_schema, ignore_errors=True)
    shutil.rmtree(gpu_schema, ignore_errors=True)

    # Compare allelic profiles
    blast_alleles = read_file_lines(os.path.join(blast_dir, 'results_alleles.tsv'))
    gpu_alleles = read_file_lines(os.path.join(gpu_dir, 'results_alleles.tsv'))

    alleles_match = blast_alleles is not None and gpu_alleles is not None and blast_alleles == gpu_alleles

    # Compare hashed profiles
    blast_hash_file = find_hashed_file(blast_dir)
    gpu_hash_file = find_hashed_file(gpu_dir)
    blast_hashed = read_file_lines(blast_hash_file) if blast_hash_file else None
    gpu_hashed = read_file_lines(gpu_hash_file) if gpu_hash_file else None
    hash_match = blast_hashed is not None and gpu_hashed is not None and blast_hashed == gpu_hashed

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK: {N_SAMPLES} Listeria genomes, {CPU_CORES} threads")
    print(f"  BLAST: {blast_time:.1f}s")
    print(f"  GPU:   {gpu_time:.1f}s")
    if blast_time > 0 and gpu_time > 0:
        print(f"  Speedup: {blast_time/gpu_time:.1f}x")
    print(f"  Allelic profiles: {'IDENTICAL' if alleles_match else 'DIFFER'}")
    print(f"  CRC32 hashed:     {'IDENTICAL' if hash_match else 'DIFFER' if blast_hashed else 'N/A'}")
    if not alleles_match and blast_alleles and gpu_alleles:
        ndiff = sum(1 for b, g in zip(blast_alleles[1:], gpu_alleles[1:]) if b != g)
        print(f"  Samples with differences: {ndiff}/{len(blast_alleles)-1}")
    print(f"  BLAST output: {blast_dir}")
    print(f"  GPU output:   {gpu_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
