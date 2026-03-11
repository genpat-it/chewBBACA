#!/usr/bin/env python3
"""
Benchmark: GPU vs BLAST on BeONE datasets.

Downloads genome assemblies from Zenodo and cgMLST/wgMLST schemas from
Chewie-NS, then runs chewBBACA allele calling with both BLAST and GPU
backends, comparing results for determinism.

BeONE project: https://onehealthejp.eu/projects/foodborne-zoonoses/jrp-beone
Zenodo datasets: https://zenodo.org/records/7802702 (Lm), 7802723 (Se),
                  7802728 (Ec), 7802717 (Cj)

Usage:
    python benchmark_beone.py [options]

    # Run all organisms
    python benchmark_beone.py

    # Run only Listeria with 100 genomes
    python benchmark_beone.py --organism lm --n-samples 100

    # Run only GPU (skip BLAST comparison)
    python benchmark_beone.py --gpu-only
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

# ── BeONE datasets ──────────────────────────────────────────────────────────

DATASETS = {
    'lm': {
        'name': 'Listeria monocytogenes',
        'short': 'Lm',
        'zenodo_url': 'https://zenodo.org/api/records/7802702/files/BeONE_Lm_assemblies.zip/content',
        'zenodo_zip': 'BeONE_Lm_assemblies.zip',
        'chewie_sp': '18',
        'chewie_sc': '1',
        'schema_type': 'cgMLST',
    },
    'se': {
        'name': 'Salmonella enterica',
        'short': 'Se',
        'zenodo_url': 'https://zenodo.org/api/records/7802723/files/BeONE_Se_assemblies.zip/content',
        'zenodo_zip': 'BeONE_Se_assemblies.zip',
        'chewie_sp': '14',
        'chewie_sc': '1',
        'schema_type': 'wgMLST',
    },
    'ec': {
        'name': 'Escherichia coli',
        'short': 'Ec',
        'zenodo_url': 'https://zenodo.org/api/records/7802728/files/BeONE_Ec_assemblies.zip/content',
        'zenodo_zip': 'BeONE_Ec_assemblies.zip',
        'chewie_sp': '10',
        'chewie_sc': '1',
        'schema_type': 'wgMLST',
    },
    'cj': {
        'name': 'Campylobacter jejuni',
        'short': 'Cj',
        'zenodo_url': 'https://zenodo.org/api/records/7802717/files/BeONE_Cj_assemblies.zip/content',
        'zenodo_zip': 'BeONE_Cj_assemblies.zip',
        'chewie_sp': '6',
        'chewie_sc': '1',
        'schema_type': 'wgMLST',
    },
}

# ── Utilities ────────────────────────────────────────────────────────────────

def download_file(url, dest_path):
    """Download a file with progress. Tries wget, then curl."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"    Already downloaded: {os.path.basename(dest_path)}")
        return
    print(f"    Downloading {os.path.basename(dest_path)} ...")
    tmp_path = dest_path + ".part"
    try:
        subprocess.run(["wget", "-O", tmp_path, url], check=True)
    except FileNotFoundError:
        subprocess.run(["curl", "-L", "--progress-bar", "-o", tmp_path, url],
                        check=True)
    os.rename(tmp_path, dest_path)
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"    Done ({size_mb:.0f} MB)")


def _count_fastas(directory):
    """Count FASTA files in a directory."""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory)
                if f.endswith(('.fasta', '.fa', '.fna'))])


def _find_schema_dir(base_dir):
    """Find the actual schema directory (may be nested after DownloadSchema)."""
    if not os.path.exists(base_dir):
        return base_dir
    if any(f.endswith('.fasta') for f in os.listdir(base_dir)):
        return base_dir
    for d in sorted(os.listdir(base_dir)):
        candidate = os.path.join(base_dir, d)
        if not os.path.isdir(candidate):
            continue
        if any(f.endswith('.fasta') for f in os.listdir(candidate)):
            return candidate
        for d2 in sorted(os.listdir(candidate)):
            candidate2 = os.path.join(candidate, d2)
            if os.path.isdir(candidate2):
                if any(f.endswith('.fasta') for f in os.listdir(candidate2)):
                    return candidate2
    return base_dir


# ── Download functions ───────────────────────────────────────────────────────

def download_genomes(dataset, data_dir):
    """Download and extract genome assemblies from Zenodo."""
    genomes_dir = os.path.join(data_dir, "genomes")
    if _count_fastas(genomes_dir) > 0:
        print(f"    Genomes already present: {_count_fastas(genomes_dir)} FASTA files")
        return genomes_dir

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, dataset['zenodo_zip'])
    download_file(dataset['zenodo_url'], zip_path)

    print("    Extracting assemblies...")
    extract_tmp = os.path.join(data_dir, "_extract_tmp")
    os.makedirs(extract_tmp, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_tmp)

    os.makedirs(genomes_dir, exist_ok=True)
    for ext in ('*.fasta', '*.fa', '*.fna'):
        for f in glob.glob(os.path.join(extract_tmp, "**", ext), recursive=True):
            shutil.move(f, os.path.join(genomes_dir, os.path.basename(f)))
    shutil.rmtree(extract_tmp, ignore_errors=True)

    n = _count_fastas(genomes_dir)
    if n == 0:
        print("    ERROR: No FASTA files found in the archive!")
        sys.exit(1)
    print(f"    Ready: {n} genome assemblies")
    return genomes_dir


def download_schema(dataset, data_dir):
    """Download schema from Chewie-NS."""
    schema_base = os.path.join(data_dir, "schema")
    if os.path.exists(schema_base):
        schema_dir = _find_schema_dir(schema_base)
        n = len([f for f in os.listdir(schema_dir) if f.endswith('.fasta')])
        if n > 0:
            print(f"    Schema already present: {n} loci ({dataset['schema_type']})")
            return schema_dir

    sp, sc = dataset['chewie_sp'], dataset['chewie_sc']
    print(f"    Downloading {dataset['schema_type']} schema from Chewie-NS (sp={sp}, sc={sc})...")
    try:
        subprocess.run([
            sys.executable, "-m", "CHEWBBACA.chewBBACA",
            "DownloadSchema", "-sp", sp, "-sc", sc, "-o", schema_base,
        ], check=True)
    except subprocess.CalledProcessError:
        print(f"\n    ERROR: Failed to download schema.")
        print(f"    Try manually: chewBBACA.py DownloadSchema -sp {sp} -sc {sc} -o {schema_base}")
        sys.exit(1)

    schema_dir = _find_schema_dir(schema_base)
    n = len([f for f in os.listdir(schema_dir) if f.endswith('.fasta')])
    print(f"    Ready: {n} loci")
    return schema_dir


# ── Allele calling ───────────────────────────────────────────────────────────

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
    print(f"    Genomes: {len(genome_files)}")

    # Build loci list
    loci_list = os.path.join(output_dir, 'loci.txt')
    schema_files = [os.path.join(schema_dir, f)
                    for f in os.listdir(schema_dir)
                    if f.endswith('.fasta')]
    with open(loci_list, 'w') as f:
        f.write('\n'.join(schema_files))
    print(f"    Loci: {len(schema_files)}")

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


# ── Result comparison ────────────────────────────────────────────────────────

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


def save_results(output_dir, results_dir, organism_key):
    """Copy allelic profiles and hashed profiles to results directory."""
    os.makedirs(results_dir, exist_ok=True)

    alleles_src = os.path.join(output_dir, 'results_alleles.tsv')
    if os.path.exists(alleles_src):
        shutil.copy2(alleles_src, os.path.join(results_dir,
                     f'{organism_key}_results_alleles.tsv'))

    hashed_file = find_hashed_file(output_dir)
    if hashed_file:
        shutil.copy2(hashed_file, os.path.join(results_dir,
                     f'{organism_key}_results_alleles_hashed.tsv'))


# ── Per-organism benchmark ───────────────────────────────────────────────────

def run_organism_benchmark(key, dataset, args):
    """Run full benchmark for one organism. Returns result dict."""
    name = dataset['name']
    data_dir = os.path.join(args.data_dir, key)
    output_dir = os.path.join(args.output_dir, key)
    os.makedirs(output_dir, exist_ok=True)

    n_label = args.n_samples or "all"
    print(f"\n{'=' * 70}")
    print(f"  {name} ({dataset['short']}) — {n_label} genomes, {args.cpu_cores} threads")
    print(f"{'=' * 70}")

    # Download
    if not args.skip_download:
        print(f"\n  Downloading assemblies...")
        genomes_dir = download_genomes(dataset, data_dir)
        print(f"\n  Downloading schema...")
        schema_dir = download_schema(dataset, data_dir)
    else:
        genomes_dir = os.path.join(data_dir, "genomes")
        schema_base = os.path.join(data_dir, "schema")
        if _count_fastas(genomes_dir) == 0:
            print(f"  ERROR: No genomes in {genomes_dir}. Run without --skip-download.")
            return None
        schema_dir = _find_schema_dir(schema_base)

    n_genomes = _count_fastas(genomes_dir)
    if args.n_samples:
        n_genomes = min(n_genomes, args.n_samples)
    n_loci = len([f for f in os.listdir(schema_dir) if f.endswith('.fasta')])

    result = {
        'name': name, 'short': dataset['short'], 'key': key,
        'n_genomes': n_genomes, 'n_loci': n_loci,
        'schema_type': dataset['schema_type'],
        'blast_time': 0, 'gpu_time': 0,
        'alleles_match': None, 'hash_match': None, 'n_differ': 0,
    }

    blast_dir = None
    gpu_dir = None

    # BLAST run
    if not args.gpu_only:
        blast_schema = tempfile.mkdtemp(prefix=f'schema_{key}_blast_', dir=output_dir)
        shutil.copytree(schema_dir, os.path.join(blast_schema, 'schema'),
                        dirs_exist_ok=True)
        print(f"\n  Running BLAST...")
        blast_dir = os.path.join(output_dir, "blast_results")
        os.makedirs(blast_dir, exist_ok=True)
        t0 = time.time()
        try:
            run_allele_call(blast_dir, os.path.join(blast_schema, 'schema'),
                            genomes_dir, args.n_samples, args.cpu_cores,
                            use_gpu=False)
        except Exception as e:
            print(f"    BLAST failed: {e}")
            import traceback; traceback.print_exc()
        result['blast_time'] = time.time() - t0
        print(f"    BLAST time: {result['blast_time']:.1f}s")
        shutil.rmtree(blast_schema, ignore_errors=True)

    # GPU run
    if not args.blast_only:
        gpu_schema = tempfile.mkdtemp(prefix=f'schema_{key}_gpu_', dir=output_dir)
        shutil.copytree(schema_dir, os.path.join(gpu_schema, 'schema'),
                        dirs_exist_ok=True)
        print(f"\n  Running GPU...")
        gpu_dir = os.path.join(output_dir, "gpu_results")
        os.makedirs(gpu_dir, exist_ok=True)
        t0 = time.time()
        try:
            run_allele_call(gpu_dir, os.path.join(gpu_schema, 'schema'),
                            genomes_dir, args.n_samples, args.cpu_cores,
                            use_gpu=True)
        except Exception as e:
            print(f"    GPU failed: {e}")
            import traceback; traceback.print_exc()
        result['gpu_time'] = time.time() - t0
        print(f"    GPU time: {result['gpu_time']:.1f}s")
        shutil.rmtree(gpu_schema, ignore_errors=True)

    # Compare
    if blast_dir and gpu_dir:
        blast_alleles = read_file_lines(os.path.join(blast_dir, 'results_alleles.tsv'))
        gpu_alleles = read_file_lines(os.path.join(gpu_dir, 'results_alleles.tsv'))
        result['alleles_match'] = (blast_alleles is not None and
                                   gpu_alleles is not None and
                                   blast_alleles == gpu_alleles)
        if not result['alleles_match'] and blast_alleles and gpu_alleles:
            result['n_differ'] = sum(1 for b, g in zip(blast_alleles[1:],
                                     gpu_alleles[1:]) if b != g)

        blast_hashed = read_file_lines(find_hashed_file(blast_dir)) if find_hashed_file(blast_dir) else None
        gpu_hashed = read_file_lines(find_hashed_file(gpu_dir)) if find_hashed_file(gpu_dir) else None
        result['hash_match'] = (blast_hashed is not None and
                                gpu_hashed is not None and
                                blast_hashed == gpu_hashed)

    # Save profiles to results/
    results_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    if blast_dir:
        save_results(blast_dir, results_save_dir, f'{key}_blast')
    if gpu_dir:
        save_results(gpu_dir, results_save_dir, f'{key}_gpu')

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GPU vs BLAST chewBBACA on BeONE datasets"
    )
    parser.add_argument("--data-dir", default="./beone_data",
                        help="Directory for downloaded data (default: ./beone_data)")
    parser.add_argument("--output-dir", default="./beone_benchmark",
                        help="Directory for benchmark outputs (default: ./beone_benchmark)")
    parser.add_argument("--organism", choices=list(DATASETS.keys()),
                        nargs='+', default=None,
                        help="Organisms to benchmark (default: all). "
                             "Options: lm (Listeria), se (Salmonella), "
                             "ec (E. coli), cj (Campylobacter)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of genomes per organism (default: all)")
    parser.add_argument("--cpu-cores", type=int, default=8,
                        help="CPU threads for BLAST (default: 8)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download step")
    parser.add_argument("--gpu-only", action="store_true",
                        help="Run only GPU version")
    parser.add_argument("--blast-only", action="store_true",
                        help="Run only BLAST version")
    args = parser.parse_args()

    args.data_dir = os.path.abspath(args.data_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    organisms = args.organism or list(DATASETS.keys())

    print("=" * 70)
    print("  BeONE BENCHMARK — GPU vs BLAST chewBBACA")
    print(f"  Organisms: {', '.join(DATASETS[k]['name'] for k in organisms)}")
    print(f"  Samples: {args.n_samples or 'all'} per organism")
    print(f"  CPU threads: {args.cpu_cores}")
    print("=" * 70)

    results = []
    for key in organisms:
        r = run_organism_benchmark(key, DATASETS[key], args)
        if r:
            results.append(r)

    # Summary table
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Organism':<30} {'Genomes':>7} {'Loci':>6} {'BLAST':>8} {'GPU':>8} {'Speed':>6} {'CRC32':>10}")
    print(f"  {'-'*30} {'-'*7} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*10}")
    for r in results:
        speedup = f"{r['blast_time']/r['gpu_time']:.1f}x" if r['blast_time'] > 0 and r['gpu_time'] > 0 else "N/A"
        if r['hash_match'] is True:
            crc = "IDENTICAL"
        elif r['hash_match'] is False:
            crc = "DIFFER"
        else:
            crc = "N/A"
        bt = f"{r['blast_time']:.0f}s" if r['blast_time'] > 0 else "N/A"
        gt = f"{r['gpu_time']:.0f}s" if r['gpu_time'] > 0 else "N/A"
        print(f"  {r['name']:<30} {r['n_genomes']:>7} {r['n_loci']:>6} {bt:>8} {gt:>8} {speedup:>6} {crc:>10}")

    print(f"\n  Results saved to: results/")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
