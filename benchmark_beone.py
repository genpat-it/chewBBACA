#!/usr/bin/env python3
"""
Benchmark: GPU vs BLAST on BeONE Listeria monocytogenes dataset.

Downloads the BeONE L. monocytogenes assemblies from Zenodo and the cgMLST
schema from Chewie-NS, then runs chewBBACA allele calling with both BLAST
and GPU backends, comparing results for determinism.

BeONE project: https://onehealthejp.eu/projects/foodborne-zoonoses/jrp-beone

Usage:
    python benchmark_beone.py [options]

Options:
    --data-dir DIR      Directory to store downloaded data (default: ./beone_data)
    --output-dir DIR    Directory for benchmark outputs (default: ./beone_benchmark)
    --n-samples N       Number of genomes to use (default: all available)
    --cpu-cores N       CPU threads for BLAST (default: 8)
    --skip-download     Skip download if data already exists
    --gpu-only          Run only GPU version (skip BLAST)
    --blast-only        Run only BLAST version (skip GPU)
"""

import argparse
import os
import sys
import time
import shutil
import subprocess
import tempfile
import zipfile
import glob

sys.path.insert(0, os.path.dirname(__file__))

# BeONE Listeria monocytogenes assemblies on Zenodo
# https://zenodo.org/records/7802702
ZENODO_LM_URL = "https://zenodo.org/api/records/7802702/files/BeONE_Lm_assemblies.zip/content"

# Chewie-NS cgMLST schema for L. monocytogenes
CHEWIE_NS_SCHEMA_ID = "6"  # L. monocytogenes cgMLST
CHEWIE_NS_URL = "https://chewbbaca.online/api/NS/api/species/9/schemas/1/zip"


def download_file(url, dest_path):
    """Download a file with progress bar. Tries wget, then curl."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"  Already downloaded: {os.path.basename(dest_path)}")
        return
    print(f"  Downloading {os.path.basename(dest_path)} ...")
    tmp_path = dest_path + ".part"
    try:
        subprocess.run(["wget", "-O", tmp_path, url], check=True)
    except FileNotFoundError:
        subprocess.run(["curl", "-L", "--progress-bar", "-o", tmp_path, url],
                        check=True)
    os.rename(tmp_path, dest_path)
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"  Done ({size_mb:.0f} MB)")


def _count_fastas(directory):
    """Count FASTA files in a directory."""
    return len([f for f in os.listdir(directory)
                if f.endswith(('.fasta', '.fa', '.fna'))])


def download_beone_data(data_dir):
    """Download BeONE L. monocytogenes assemblies from Zenodo."""
    os.makedirs(data_dir, exist_ok=True)
    genomes_dir = os.path.join(data_dir, "genomes")

    if os.path.exists(genomes_dir) and _count_fastas(genomes_dir) > 0:
        print(f"  Genomes already present: {_count_fastas(genomes_dir)} FASTA files")
        return genomes_dir

    zip_path = os.path.join(data_dir, "BeONE_Lm_assemblies.zip")
    download_file(ZENODO_LM_URL, zip_path)

    print("  Extracting assemblies...")
    extract_tmp = os.path.join(data_dir, "_extract_tmp")
    os.makedirs(extract_tmp, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_tmp)

    # Flatten: move all FASTA files to genomes_dir regardless of zip structure
    os.makedirs(genomes_dir, exist_ok=True)
    for ext in ('*.fasta', '*.fa', '*.fna'):
        for f in glob.glob(os.path.join(extract_tmp, "**", ext), recursive=True):
            shutil.move(f, os.path.join(genomes_dir, os.path.basename(f)))
    shutil.rmtree(extract_tmp, ignore_errors=True)

    n = _count_fastas(genomes_dir)
    if n == 0:
        print("  ERROR: No FASTA files found in the downloaded archive!")
        print(f"  Check the contents of {zip_path}")
        sys.exit(1)
    print(f"  Ready: {n} genome assemblies")
    return genomes_dir


def _find_schema_dir(base_dir):
    """Find the actual schema directory (may be nested after DownloadSchema)."""
    # Direct check
    if any(f.endswith('.fasta') for f in os.listdir(base_dir)):
        return base_dir
    # Search one level deep
    for d in os.listdir(base_dir):
        candidate = os.path.join(base_dir, d)
        if os.path.isdir(candidate):
            if any(f.endswith('.fasta') for f in os.listdir(candidate)):
                return candidate
            # Two levels deep (DownloadSchema creates species/schema/)
            for d2 in os.listdir(candidate):
                candidate2 = os.path.join(candidate, d2)
                if os.path.isdir(candidate2):
                    if any(f.endswith('.fasta') for f in os.listdir(candidate2)):
                        return candidate2
    return base_dir


def download_schema(data_dir):
    """Download L. monocytogenes cgMLST schema from Chewie-NS."""
    schema_base = os.path.join(data_dir, "schema")

    if os.path.exists(schema_base):
        schema_dir = _find_schema_dir(schema_base)
        n = len([f for f in os.listdir(schema_dir) if f.endswith('.fasta')])
        if n > 0:
            print(f"  Schema already present: {n} loci")
            return schema_dir

    print("  Downloading cgMLST schema from Chewie-NS...")
    try:
        subprocess.run([
            sys.executable, "-m", "CHEWBBACA.chewBBACA",
            "DownloadSchema",
            "-sp", "9",   # L. monocytogenes species ID
            "-sc", "1",   # Schema ID
            "-o", schema_base,
        ], check=True)
    except subprocess.CalledProcessError:
        print("\n  ERROR: Failed to download schema.")
        print("  Try manually:")
        print(f"    chewBBACA.py DownloadSchema -sp 9 -sc 1 -o {schema_base}")
        sys.exit(1)

    schema_dir = _find_schema_dir(schema_base)
    n = len([f for f in os.listdir(schema_dir) if f.endswith('.fasta')])
    print(f"  Ready: {n} loci")
    return schema_dir


def run_allele_call(output_dir, schema_dir, genomes_dir, n_samples, cpu_cores,
                    use_gpu=False):
    """Run chewBBACA allele calling."""
    from CHEWBBACA.utils import (constants as ct,
                                  blast_wrapper as bw,
                                  file_operations as fo)
    from CHEWBBACA.AlleleCall import allele_call

    bw.disable_gpu()
    if use_gpu:
        bw.enable_gpu()

    # Build genome list
    genome_list = os.path.join(output_dir, 'genomes.txt')
    all_fastas = sorted([
        f for f in os.listdir(genomes_dir)
        if f.endswith(('.fasta', '.fa', '.fna'))
    ])
    if n_samples:
        all_fastas = all_fastas[:n_samples]
    genome_files = [os.path.join(genomes_dir, f) for f in all_fastas]
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
        'CPU cores': cpu_cores,
        'BLAST path': '',
        'CDS input': False,
        'Prodigal mode': 'single',
        'Mode': 4,
    }

    ptf_files = [os.path.join(schema_dir, f)
                 for f in os.listdir(schema_dir) if f.endswith('.trn')]
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
    parser = argparse.ArgumentParser(
        description="Benchmark GPU vs BLAST chewBBACA on BeONE L. monocytogenes"
    )
    parser.add_argument("--data-dir", default="./beone_data",
                        help="Directory for downloaded data")
    parser.add_argument("--output-dir", default="./beone_benchmark",
                        help="Directory for benchmark outputs")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of genomes (default: all)")
    parser.add_argument("--cpu-cores", type=int, default=8,
                        help="CPU threads for BLAST (default: 8)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download step")
    parser.add_argument("--gpu-only", action="store_true",
                        help="Run only GPU version")
    parser.add_argument("--blast-only", action="store_true",
                        help="Run only BLAST version")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    n_label = args.n_samples or "all"
    print("=" * 70)
    print(f"BeONE BENCHMARK: {n_label} L. monocytogenes genomes, {args.cpu_cores} threads")
    print("=" * 70)

    # Download data
    if not args.skip_download:
        print("\nStep 1/3: Downloading BeONE assemblies...")
        genomes_dir = download_beone_data(data_dir)
        print("\nStep 2/3: Downloading cgMLST schema...")
        schema_dir = download_schema(data_dir)
        print("\nStep 3/3: Running benchmark...")
    else:
        genomes_dir = os.path.join(data_dir, "genomes")
        schema_base = os.path.join(data_dir, "schema")
        if not os.path.exists(genomes_dir) or _count_fastas(genomes_dir) == 0:
            print(f"ERROR: No genomes found in {genomes_dir}")
            print("Run without --skip-download first.")
            sys.exit(1)
        schema_dir = _find_schema_dir(schema_base)

    # Independent schema copies (to avoid lock conflicts)
    blast_schema = None
    gpu_schema = None
    blast_time = 0
    gpu_time = 0
    blast_dir = None
    gpu_dir = None

    if not args.gpu_only:
        blast_schema = tempfile.mkdtemp(prefix='schema_blast_', dir=output_dir)
        shutil.copytree(schema_dir, os.path.join(blast_schema, 'schema'),
                        dirs_exist_ok=True)

    if not args.blast_only:
        gpu_schema = tempfile.mkdtemp(prefix='schema_gpu_', dir=output_dir)
        shutil.copytree(schema_dir, os.path.join(gpu_schema, 'schema'),
                        dirs_exist_ok=True)

    # BLAST run
    if not args.gpu_only:
        print("\n" + "-" * 70)
        print("Running BLAST version...")
        print("-" * 70)
        blast_dir = os.path.join(output_dir, "blast_results")
        os.makedirs(blast_dir, exist_ok=True)
        t0 = time.time()
        try:
            run_allele_call(blast_dir, os.path.join(blast_schema, 'schema'),
                            genomes_dir, args.n_samples, args.cpu_cores,
                            use_gpu=False)
        except Exception as e:
            print(f"  BLAST run failed: {e}")
            import traceback; traceback.print_exc()
        blast_time = time.time() - t0
        print(f"  BLAST time: {blast_time:.1f}s")
        shutil.rmtree(blast_schema, ignore_errors=True)

    # GPU run
    if not args.blast_only:
        print("\n" + "-" * 70)
        print("Running GPU version...")
        print("-" * 70)
        gpu_dir = os.path.join(output_dir, "gpu_results")
        os.makedirs(gpu_dir, exist_ok=True)
        t0 = time.time()
        try:
            run_allele_call(gpu_dir, os.path.join(gpu_schema, 'schema'),
                            genomes_dir, args.n_samples, args.cpu_cores,
                            use_gpu=True)
        except Exception as e:
            print(f"  GPU run failed: {e}")
            import traceback; traceback.print_exc()
        gpu_time = time.time() - t0
        print(f"  GPU time: {gpu_time:.1f}s")
        shutil.rmtree(gpu_schema, ignore_errors=True)

    # Compare results
    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")

    if blast_dir and gpu_dir:
        blast_alleles = read_file_lines(os.path.join(blast_dir, 'results_alleles.tsv'))
        gpu_alleles = read_file_lines(os.path.join(gpu_dir, 'results_alleles.tsv'))
        alleles_match = (blast_alleles is not None and gpu_alleles is not None
                         and blast_alleles == gpu_alleles)

        blast_hash_file = find_hashed_file(blast_dir)
        gpu_hash_file = find_hashed_file(gpu_dir)
        blast_hashed = read_file_lines(blast_hash_file) if blast_hash_file else None
        gpu_hashed = read_file_lines(gpu_hash_file) if gpu_hash_file else None
        hash_match = (blast_hashed is not None and gpu_hashed is not None
                      and blast_hashed == gpu_hashed)

        print(f"  Genomes:          {n_label}")
        print(f"  CPU threads:      {args.cpu_cores}")
        print(f"  BLAST time:       {blast_time:.1f}s")
        print(f"  GPU time:         {gpu_time:.1f}s")
        if blast_time > 0 and gpu_time > 0:
            print(f"  Speedup:          {blast_time/gpu_time:.2f}x")
        print(f"  Allelic profiles: {'IDENTICAL' if alleles_match else 'DIFFER'}")
        print(f"  CRC32 hashed:     {'IDENTICAL' if hash_match else 'DIFFER' if blast_hashed else 'N/A'}")

        if not alleles_match and blast_alleles and gpu_alleles:
            ndiff = sum(1 for b, g in zip(blast_alleles[1:], gpu_alleles[1:])
                        if b != g)
            print(f"  Differing samples: {ndiff}/{len(blast_alleles)-1}")
    elif blast_dir:
        print(f"  BLAST time: {blast_time:.1f}s")
    elif gpu_dir:
        print(f"  GPU time:   {gpu_time:.1f}s")

    print(f"\n  Output: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
