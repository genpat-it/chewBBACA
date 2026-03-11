#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------

This module contains functions related with the execution
of the BLAST software (https://www.ncbi.nlm.nih.gov/books/NBK279690/).

When GPU mode is enabled, BLAST calls are replaced with GPU-accelerated
Smith-Waterman alignment using the same scoring parameters (BLOSUM62,
gap open=11, gap extend=1) to produce compatible results.

Code documentation
------------------
"""


import os
import sys
import shutil
import subprocess

try:
	from utils import constants as ct
except ModuleNotFoundError:
	from CHEWBBACA.utils import constants as ct


# Load C k-mer filter extension
import ctypes as _ctypes
import numpy as _np

_KMER_FILTER_LIB = None

def _load_kmer_filter():
	"""Load the compiled C k-mer filter shared library."""
	global _KMER_FILTER_LIB
	if _KMER_FILTER_LIB is not None:
		return _KMER_FILTER_LIB
	so_path = os.path.join(os.path.dirname(__file__), 'kmer_filter.so')
	if os.path.exists(so_path):
		_KMER_FILTER_LIB = _ctypes.CDLL(so_path)
		_KMER_FILTER_LIB.kmer_filter_pairs.restype = _ctypes.c_int
		_KMER_FILTER_LIB.kmer_filter_pairs.argtypes = [
			_ctypes.c_char_p, _ctypes.POINTER(_ctypes.c_int), _ctypes.c_int,
			_ctypes.c_char_p, _ctypes.POINTER(_ctypes.c_int), _ctypes.c_int,
			_ctypes.POINTER(_ctypes.c_int), _ctypes.POINTER(_ctypes.c_int), _ctypes.c_int,
		]
	return _KMER_FILTER_LIB


def _c_kmer_filter(q_ids_list, query_records, t_ids_list, target_records):
	"""Use C extension for fast 6-mer pre-filtering."""
	lib = _load_kmer_filter()
	if lib is None:
		return None  # fallback to Python

	# Concatenate query sequences
	q_seqs_list = [query_records[qid] for qid in q_ids_list]
	q_concat = ''.join(q_seqs_list).encode('ascii')
	q_offsets = _np.zeros(len(q_ids_list) + 1, dtype=_np.int32)
	off = 0
	for i, s in enumerate(q_seqs_list):
		q_offsets[i] = off
		off += len(s)
	q_offsets[len(q_ids_list)] = off

	# Concatenate target sequences
	t_seqs_list = [target_records[tid] for tid in t_ids_list]
	t_concat = ''.join(t_seqs_list).encode('ascii')
	t_offsets = _np.zeros(len(t_ids_list) + 1, dtype=_np.int32)
	off = 0
	for i, s in enumerate(t_seqs_list):
		t_offsets[i] = off
		off += len(s)
	t_offsets[len(t_ids_list)] = off

	# Allocate output (generous upper bound)
	max_pairs = len(q_ids_list) * min(len(t_ids_list), 500)
	out_qidx = _np.zeros(max_pairs, dtype=_np.int32)
	out_tidx = _np.zeros(max_pairs, dtype=_np.int32)

	n_pairs = lib.kmer_filter_pairs(
		q_concat,
		q_offsets.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int)),
		len(q_ids_list),
		t_concat,
		t_offsets.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int)),
		len(t_ids_list),
		out_qidx.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int)),
		out_tidx.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int)),
		max_pairs,
	)
	if n_pairs < 0:
		return None  # overflow, fallback

	pairs = []
	for i in range(n_pairs):
		pairs.append((q_ids_list[out_qidx[i]], t_ids_list[out_tidx[i]]))
	return pairs


# GPU mode flag - set by enable_gpu() or --gpu CLI flag
_GPU_MODE = False
_GPU_ALIGNER = None
_GPU_DEVICE = 0


def enable_gpu(device=0):
	"""Enable GPU-accelerated alignment mode (lazy init).

	The GPU aligner is created on first use, not here. This allows
	CPU-only multiprocessing steps (CDS prediction, hashing) to fork
	safely before the CUDA context is created.
	"""
	global _GPU_MODE, _GPU_DEVICE
	_GPU_MODE = True
	_GPU_DEVICE = device
	print(f'GPU mode enabled (device {device}, lazy init)')


def _get_gpu_aligner():
	"""Get or create the GPU aligner (lazy initialization)."""
	global _GPU_ALIGNER
	if _GPU_ALIGNER is None:
		try:
			from CHEWBBACA.utils.gpu_sw import GPUAligner
		except (ImportError, ModuleNotFoundError):
			try:
				from utils.gpu_sw import GPUAligner
			except ImportError:
				raise ImportError(
					"GPU mode requires CuPy. Install with: pip install cupy-cuda12x")
		_GPU_ALIGNER = GPUAligner(device=_GPU_DEVICE)
	return _GPU_ALIGNER


def disable_gpu():
	"""Disable GPU mode and fall back to BLAST."""
	global _GPU_MODE, _GPU_ALIGNER
	_GPU_MODE = False
	_GPU_ALIGNER = None


def is_gpu_enabled():
	"""Check if GPU mode is currently enabled."""
	return _GPU_MODE


def make_blast_db(makeblastdb_path, input_fasta, output_path, db_type):
	"""Create a BLAST database.

	In GPU mode, stores a reference to the source FASTA file instead
	of creating a BLAST database.

	Parameters
	----------
	makeblastdb_path : str
		Path to the 'makeblastdb' executable.
	input_fasta : str
		Path to the FASTA file that contains the sequences that
		will be added to the BLAST database.
	output_path : str
		Path to the directory where the database files will be
		created. Database files will have the same basename as
		the `input_fasta`.
	db_type : str
		Type of the database, nucleotide (nuc) or protein (prot).

	Returns
	-------
	stdout : bytes
		BLAST stdout.
	stderr : bytes or str
		BLAST stderr.
	"""
	if _GPU_MODE:
		# Copy FASTA file so GPU can read it even if the original is deleted
		gpu_fasta_copy = output_path + '.gpu.fasta'
		shutil.copy2(input_fasta, gpu_fasta_copy)
		marker_file = output_path + '.gpu_fasta_path'
		with open(marker_file, 'w') as f:
			f.write(os.path.abspath(gpu_fasta_copy))
		return [b'GPU mode: BLAST DB not needed', b'']

	# Original BLAST implementation
	makedb_cmd = [makeblastdb_path, '-in', input_fasta,
				  '-out', output_path, '-parse_seqids',
				  '-dbtype', db_type, '-blastdb_version', '5']

	makedb_process = subprocess.Popen(makedb_cmd,
									  stdout=subprocess.PIPE,
									  stderr=subprocess.PIPE)

	stdout, stderr = makedb_process.communicate()

	if len(stderr) > 0:
		sys.exit(f'Could not create BLAST database for {input_fasta}\n'
				 f'{makeblastdb_path} returned the following stderr:\n{stderr}')

	return [stdout, stderr]


def determine_blast_task(sequences, blast_type='blastp'):
	"""Determine the type of BLAST task to execute.

	In GPU mode, the same Smith-Waterman algorithm handles all sequence
	lengths, so this is kept for interface compatibility.

	Parameters
	----------
	sequences : list
		List that contains strings representing DNA or
		protein sequences.
	blast_type : str
		Used to define the type of application, 'blastn'
		or 'blastp'.

	Returns
	-------
	blast_task : str
		A string that indicates the type of BLAST task to
		execute based on the minimum sequence size.
	"""
	length_threshold = ct.BLAST_TASK_THRESHOLD[blast_type]
	sequence_lengths = [len(p) for p in sequences]
	minimum_length = min(sequence_lengths)
	if minimum_length < length_threshold:
		blast_task = '{0}-short'.format(blast_type)
	else:
		blast_task = blast_type

	return blast_task


def run_blast(blast_path, blast_db, fasta_file, blast_output,
			  max_hsps=1, threads=1, ids_file=None, blast_task=None,
			  max_targets=None, composition_stats=None):
	"""Execute BLAST (or GPU Smith-Waterman) to align sequences.

	In GPU mode, reads sequences from FASTA files and runs
	Smith-Waterman on GPU with BLOSUM62 scoring.

	Parameters
	----------
	blast_path : str
		Path to the BLAST application executable.
	blast_db : str
		Path to the BLAST database.
	fasta_file : str
		Path to the FASTA file with sequences to align against
		the BLAST database.
	blast_output : str
		Path to the file that will be created to store the
		results.
	max_hsps : int
		Maximum number of High Scoring Pairs per pair of aligned
		sequences.
	threads : int
		Number of threads/cores used to run BLAST.
	ids_file : str
		Path to a file with sequence identifiers, one per line.
	blast_task : str
		Type of BLAST task.
	max_targets : int
		Maximum number of target/subject sequences to align against.
	composition_stats : int
		Composition-based statistics method used by BLAST.

	Returns
	-------
	stdout : bytes
		BLAST stdout.
	stderr : bytes or str
		BLAST stderr.
	"""
	if _GPU_MODE:
		return _gpu_run_blast(blast_db, fasta_file, blast_output,
							  max_hsps, ids_file, max_targets)

	# Original BLAST implementation
	blast_args = [blast_path, '-db', blast_db, '-query', fasta_file,
				  '-out', blast_output, '-outfmt', ct.BLAST_DEFAULT_OUTFMT,
				  '-max_hsps', str(max_hsps), '-num_threads', str(threads),
				  '-evalue', '0.001',
				  '-comp_based_stats', '0']

	if ids_file is not None:
		blast_args.extend(['-seqidlist', ids_file])
	if blast_task is not None:
		blast_args.extend(['-task', blast_task])
	if max_targets is not None:
		blast_args.extend(['-max_target_seqs', str(max_targets)])
	if composition_stats is not None:
		blast_args.extend(['-comp_based_stats', str(composition_stats)])

	blast_process = subprocess.Popen(blast_args,
								  stdout=subprocess.PIPE,
								  stderr=subprocess.PIPE)

	stdout, stderr = blast_process.communicate()

	if len(stderr) > 0:
		sys.exit(f'Error while running BLASTp for {fasta_file}\n'
				 f'{blast_path} returned the following error:\n{stderr}')

	return [stdout, stderr]


def _gpu_run_blast(blast_db, fasta_file, blast_output,
				   max_hsps=1, ids_file=None, max_targets=None):
	"""GPU implementation of run_blast using Smith-Waterman.

	Reads query and target sequences, runs SW on GPU, and writes
	results in BLAST tabular format.
	"""
	return _gpu_run_blast_impl(blast_db, fasta_file, blast_output, max_hsps, ids_file, max_targets)

# Cache for target sequences across multiple GPU BLAST calls
_TARGET_CACHE = {'db': None, 'ids_file': None, 'records': None, 'encoded': None, 'kmer_index': None}


def _gpu_run_blast_impl(blast_db, fasta_file, blast_output,
				   max_hsps=1, ids_file=None, max_targets=None):
	"""Actual GPU BLAST implementation with optimized pair building."""
	from Bio import SeqIO
	from CHEWBBACA.utils.gpu_sw import encode_sequence
	import numpy as np
	import time as _time
	_t0 = _time.time()

	# Load query sequences
	query_records = {}
	if os.path.exists(fasta_file):
		for record in SeqIO.parse(fasta_file, 'fasta'):
			query_records[record.id] = str(record.seq)

	# Load target sequences (cached across calls with same blast_db + ids_file)
	cache_key = (blast_db, ids_file)
	if _TARGET_CACHE['db'] == blast_db and _TARGET_CACHE['ids_file'] == ids_file:
		target_records = _TARGET_CACHE['records']
		t_ids_list = list(target_records.keys())
		t_encoded = _TARGET_CACHE['encoded']
	else:
		target_records = _gpu_load_targets(blast_db, ids_file)
		t_ids_list = list(target_records.keys())
		t_encoded = {tid: encode_sequence(target_records[tid]) for tid in t_ids_list}
		_TARGET_CACHE['db'] = blast_db
		_TARGET_CACHE['ids_file'] = ids_file
		_TARGET_CACHE['records'] = target_records
		_TARGET_CACHE['encoded'] = t_encoded
		_TARGET_CACHE['kmer_index'] = None  # rebuild on next use

	if len(query_records) == 0 or len(target_records) == 0:
		with open(blast_output, 'w') as f:
			pass
		return [b'', b'']

	# Encode query sequences
	q_ids_list = list(query_records.keys())
	q_encoded = {qid: encode_sequence(query_records[qid]) for qid in q_ids_list}

	_t_load = _time.time() - _t0

	# Build pairs with 6-mer pre-filter
	_t_filter_start = _time.time()
	total_possible = len(q_ids_list) * len(t_ids_list)
	pairs = []

	if total_possible > 50000:
		# Try C extension first (much faster), fall back to Python
		c_pairs = _c_kmer_filter(q_ids_list, query_records, t_ids_list, target_records)
		if c_pairs is not None:
			pairs = c_pairs
		else:
			# Python fallback
			KMER_SIZE = 6
			MIN_SHARED_KMERS = 2
			n_targets = len(t_ids_list)
			if _TARGET_CACHE['kmer_index'] is not None:
				kmer_to_arrs = _TARGET_CACHE['kmer_index']
			else:
				tid_to_idx = {tid: i for i, tid in enumerate(t_ids_list)}
				kmer_to_idxs = {}
				for tid in t_ids_list:
					tseq = target_records[tid]
					tidx = tid_to_idx[tid]
					for i in range(len(tseq) - KMER_SIZE + 1):
						kmer = tseq[i:i+KMER_SIZE]
						if kmer not in kmer_to_idxs:
							kmer_to_idxs[kmer] = []
						kmer_to_idxs[kmer].append(tidx)
				kmer_to_arrs = {k: np.array(v, dtype=np.int32) for k, v in kmer_to_idxs.items()}
				_TARGET_CACHE['kmer_index'] = kmer_to_arrs
			for qid in q_ids_list:
				qseq = query_records[qid]
				qlen = len(qseq)
				if qlen < KMER_SIZE:
					continue
				hit_arrays = []
				seen_kmers = set()
				for i in range(qlen - KMER_SIZE + 1):
					kmer = qseq[i:i+KMER_SIZE]
					if kmer not in seen_kmers:
						seen_kmers.add(kmer)
						arr = kmer_to_arrs.get(kmer)
						if arr is not None:
							hit_arrays.append(arr)
				if len(hit_arrays) < MIN_SHARED_KMERS:
					continue
				combined = np.concatenate(hit_arrays)
				counts = np.bincount(combined, minlength=n_targets)
				matching_idxs = np.where(counts >= MIN_SHARED_KMERS)[0]
				for idx in matching_idxs:
					pairs.append((qid, t_ids_list[idx]))
	else:
		for qid in q_ids_list:
			for tid in t_ids_list:
				pairs.append((qid, tid))

	if len(pairs) == 0:
		with open(blast_output, 'w') as f:
			pass
		return [b'', b'']

	_t_filter = _time.time() - _t_filter_start

	# Build flattened arrays for GPU
	_t_build_start = _time.time()
	num_pairs = len(pairs)
	q_len_lookup = {qid: len(enc) for qid, enc in q_encoded.items()}
	t_len_lookup = {tid: len(enc) for tid, enc in t_encoded.items()}
	q_lengths = np.array([q_len_lookup[p[0]] for p in pairs], dtype=np.int32)
	t_lengths = np.array([t_len_lookup[p[1]] for p in pairs], dtype=np.int32)

	q_offsets = np.zeros(num_pairs, dtype=np.int32)
	t_offsets = np.zeros(num_pairs, dtype=np.int32)
	if num_pairs > 1:
		q_offsets[1:] = np.cumsum(q_lengths[:-1])
		t_offsets[1:] = np.cumsum(t_lengths[:-1])

	q_total = int(q_offsets[-1] + q_lengths[-1]) if num_pairs > 0 else 0
	t_total = int(t_offsets[-1] + t_lengths[-1]) if num_pairs > 0 else 0
	all_queries = np.empty(q_total, dtype=np.int32)
	all_targets = np.empty(t_total, dtype=np.int32)
	for i, (qid, tid) in enumerate(pairs):
		qo, ql = int(q_offsets[i]), int(q_lengths[i])
		to, tl = int(t_offsets[i]), int(t_lengths[i])
		all_queries[qo:qo+ql] = q_encoded[qid]
		all_targets[to:to+tl] = t_encoded[tid]

	_t_build = _time.time() - _t_build_start

	# Run GPU alignment
	_t_gpu_start = _time.time()
	q_ids_arr = [p[0] for p in pairs]
	t_ids_arr = [p[1] for p in pairs]
	all_results = _get_gpu_aligner().align_pairs_raw(
		all_queries, all_targets, q_offsets, t_offsets,
		q_lengths, t_lengths, q_ids_arr, t_ids_arr
	)

	_t_gpu = _time.time() - _t_gpu_start

	# Apply max_hsps: keep only best score per query-target pair
	if max_hsps == 1:
		best_per_pair = {}
		for r in all_results:
			key = (r[0], r[4])
			score = float(r[6])
			if key not in best_per_pair or score > float(best_per_pair[key][6]):
				best_per_pair[key] = r
		all_results = list(best_per_pair.values())

	# Apply max_targets: keep only top N targets per query
	if max_targets is not None:
		from collections import defaultdict
		by_query = defaultdict(list)
		for r in all_results:
			by_query[r[0]].append(r)
		all_results = []
		for qid, hits in by_query.items():
			hits.sort(key=lambda x: float(x[6]), reverse=True)
			all_results.extend(hits[:max_targets])


	# Write BLAST tabular output
	with open(blast_output, 'w') as f:
		for r in all_results:
			f.write('\t'.join(str(x) for x in r) + '\n')

	_t_total = _time.time() - _t0
	print(f'  [GPU-BLAST] queries={len(query_records)} targets={len(target_records)} pairs={num_pairs} | load={_t_load:.2f}s filter={_t_filter:.2f}s build={_t_build:.2f}s gpu={_t_gpu:.2f}s total={_t_total:.2f}s')

	return [b'', b'']


def _gpu_load_targets(blast_db, ids_file=None):
	"""Load target sequences for GPU alignment.

	Finds the source FASTA via the .gpu_fasta_path marker file
	and optionally filters by sequence IDs.
	"""
	from Bio import SeqIO

	fasta_path = None

	# Check GPU marker file
	marker_file = blast_db + '.gpu_fasta_path'
	if os.path.exists(marker_file):
		with open(marker_file, 'r') as f:
			fasta_path = f.read().strip()

	# Fallback: try common extensions
	if fasta_path is None or not os.path.exists(fasta_path):
		for ext in ['.fasta', '.fa', '.faa', '.fna']:
			candidate = blast_db + ext
			if os.path.exists(candidate):
				fasta_path = candidate
				break

	if fasta_path is None or not os.path.exists(fasta_path):
		return {}

	# Load sequences (strip 'lcl|' prefix that BLAST would strip automatically)
	target_records = {}
	for record in SeqIO.parse(fasta_path, 'fasta'):
		rid = record.id
		if rid.startswith('lcl|'):
			rid = rid[4:]
		target_records[rid] = str(record.seq)

	# Filter by IDs if provided
	if ids_file is not None:
		allowed_ids = _gpu_load_seqids(ids_file)
		if allowed_ids is not None:
			target_records = {k: v for k, v in target_records.items()
							  if k in allowed_ids}

	return target_records


def _gpu_load_seqids(ids_file):
	"""Load sequence IDs from text file (GPU mode reads text, not binary)."""
	if ids_file is None:
		return None

	# Try text version (without .bin suffix)
	text_file = ids_file
	if ids_file.endswith('.bin'):
		text_file = ids_file[:-4]

	if os.path.exists(text_file):
		ids = set()
		with open(text_file, 'r') as f:
			for line in f:
				line = line.strip()
				if line:
					ids.add(line)
		return ids

	return None


def run_blastdb_aliastool(blastdb_aliastool_path, seqid_infile, seqid_outfile):
	"""Convert list of sequence identifiers into binary format.

	In GPU mode, copies the text file as-is since GPU alignment
	reads text seqid files directly.

	Parameters
	----------
	blastdb_aliastool_path : str
		Path to the blastdb_aliastool executable.
	seqid_infile :  str
		Path to the file that contains the list of sequence identifiers.
	seqid_outfile : str
		Path to the output file in binary format to pass to the -seqidlist
		parameter of BLAST>=2.10.

	Returns
	-------
	stdout : bytes
		BLAST stdout.
	stderr : bytes or str
		BLAST stderr.
	"""
	if _GPU_MODE:
		# Just copy the text file - GPU reads it directly
		shutil.copy2(seqid_infile, seqid_outfile)
		return [b'', b'']

	# Original BLAST implementation
	blastdb_aliastool_args = [blastdb_aliastool_path, '-seqid_file_in',
							  seqid_infile, '-seqid_file_out', seqid_outfile]

	blastdb_aliastool_process = subprocess.Popen(blastdb_aliastool_args,
												 stdout=subprocess.PIPE,
												 stderr=subprocess.PIPE)

	stdout, stderr = blastdb_aliastool_process.communicate()

	if len(stderr) > 0:
		sys.exit(f'Could not convert {seqid_infile} to binary format.\n'
				 f'{blastdb_aliastool_path} returned the following error:\n{stderr}')

	return [stdout, stderr]


def run_blastdbcmd(blastdbcmd_path, blast_db, output_file):
	"""Run blastdbcmd to extract sequences from a BLAST database.

	In GPU mode, copies the source FASTA file.

	Parameters
	----------
	blastdbcmd_path : str
		Path to the blastdbcmd executable.
	blast_db : str
		Path to the BLAST database.
	output_file : str
		Path to the output file that will store the sequences.

	Returns
	-------
	stdout : bytes
		BLAST stdout.
	stderr : bytes or str
		BLAST stderr.
	"""
	if _GPU_MODE:
		marker_file = blast_db + '.gpu_fasta_path'
		if os.path.exists(marker_file):
			with open(marker_file, 'r') as f:
				fasta_path = f.read().strip()
			if os.path.exists(fasta_path):
				shutil.copy2(fasta_path, output_file)
				return [b'', b'']
		sys.exit(f'Could not extract sequences from {blast_db}.\n')

	# Original BLAST implementation
	blastdbcmd_args = [blastdbcmd_path, '-db', blast_db, '-out', output_file, '-entry', 'all']

	blastdbcmd_process = subprocess.Popen(blastdbcmd_args,
										  stdout=subprocess.PIPE,
										  stderr=subprocess.PIPE)

	stdout, stderr = blastdbcmd_process.communicate()

	if len(stderr) > 0:
		sys.exit(f'Cound not extract sequences from {blast_db}.\n')

	return [stdout, stderr]
