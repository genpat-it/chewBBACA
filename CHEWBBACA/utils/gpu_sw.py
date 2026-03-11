#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-accelerated Smith-Waterman alignment using CuPy.

Replaces BLAST for chewBBACA allele calling with deterministic
Smith-Waterman on NVIDIA GPUs via CUDA.

Uses BLOSUM62 scoring matrix with gap open=11, gap extend=1
(same defaults as BLASTp) to produce compatible raw scores.
"""

import os
import numpy as np

# Ensure CUDA paths are set before importing CuPy
if 'CUDA_HOME' not in os.environ:
    for cuda_path in ['/usr/local/cuda', '/usr/local/cuda-13.1',
                      '/usr/local/cuda-12', '/opt/cuda']:
        if os.path.isdir(cuda_path):
            os.environ['CUDA_HOME'] = cuda_path
            os.environ['CUDA_PATH'] = cuda_path
            break

try:
    import cupy as cp
except ImportError:
    cp = None

# BLOSUM62 scoring matrix
# Amino acid order: A R N D C Q E G H I L K M F P S T W Y V B Z X *
AA_ORDER = 'ARNDCQEGHILKMFPSTWYVBZX*'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

# fmt: off
BLOSUM62_DATA = [
#    A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   B   Z   X   *
  [  4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4],  # A
  [ -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4],  # R
  [ -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4],  # N
  [ -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4],  # D
  [  0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4],  # C
  [ -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4],  # Q
  [ -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4],  # E
  [  0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4],  # G
  [ -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4],  # H
  [ -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4],  # I
  [ -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4],  # L
  [ -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4],  # K
  [ -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4],  # M
  [ -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4],  # F
  [ -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4],  # P
  [  1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4],  # S
  [  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4],  # T
  [ -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4],  # W
  [ -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4],  # Y
  [  0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4],  # V
  [ -2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4],  # B
  [ -1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4],  # Z
  [  0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4],  # X
  [ -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1],  # *
]
# fmt: on

BLOSUM62 = np.array(BLOSUM62_DATA, dtype=np.int32)

# Gap penalties (BLAST defaults for blastp)
GAP_OPEN = 11
GAP_EXTEND = 1

# CUDA kernel for batched Smith-Waterman.
# 1 thread per block, BLOSUM62 cached in shared memory for faster access.
# 4 shared arrays: H, E (float) for DP + Hs, Es (int) for start tracking.

_SW_KERNEL_CODE = r'''
extern "C" __global__
void smith_waterman_batch(
    const int* __restrict__ queries,
    const int* __restrict__ targets,
    const int* __restrict__ query_offsets,
    const int* __restrict__ target_offsets,
    const int* __restrict__ query_lengths,
    const int* __restrict__ target_lengths,
    const int* __restrict__ blosum62,
    const int gap_open,
    const int gap_extend,
    float* __restrict__ results,
    const int num_pairs,
    const int max_query_len,
    const int max_target_len
) {
    int pair_idx = blockIdx.x;
    if (pair_idx >= num_pairs) return;
    if (threadIdx.x != 0) return;

    int qlen = query_lengths[pair_idx];
    int tlen = target_lengths[pair_idx];
    int qoff = query_offsets[pair_idx];
    int toff = target_offsets[pair_idx];

    // Shared memory layout:
    // H[stride] + E[stride] (floats) + Hs[stride] + Es[stride] (ints) + blosum[576] (ints)
    int stride = max_query_len + 1;
    extern __shared__ float shared_mem[];
    float* H  = shared_mem;
    float* E  = shared_mem + stride;
    int*   Hs = (int*)(shared_mem + 2 * stride);
    int*   Es = Hs + stride;
    int*   blosum_s = Es + stride;

    // Cache BLOSUM62 in shared memory (576 ints = 2.25 KB)
    for (int k = 0; k < 576; k++) {
        blosum_s[k] = blosum62[k];
    }

    for (int i = 0; i <= qlen; i++) {
        H[i] = 0.0f;
        E[i] = 0.0f;
        Hs[i] = i;
        Es[i] = i;
    }

    float max_score = 0.0f;
    int best_qi = 0, best_qj = 0, best_qstart = 0;
    int gap_oe = gap_open + gap_extend;

    for (int j = 1; j <= tlen; j++) {
        int tj = targets[toff + j - 1];
        const int* brow = blosum_s + tj;
        float Fval = 0.0f;
        int Fs = 0;
        float h_diag = 0.0f;
        int h_diag_s = 0;

        for (int i = 1; i <= qlen; i++) {
            int qi = queries[qoff + i - 1];
            float h_left = H[i];
            int h_left_s = Hs[i];
            float match_val = h_diag + (float)brow[qi * 24];
            int match_s = h_diag_s;

            // E: horizontal gap (gap in target)
            float e_open = h_left - gap_oe;
            float e_ext  = E[i] - gap_extend;
            if (e_open > e_ext) {
                E[i] = e_open;
                Es[i] = h_left_s;
            } else {
                E[i] = e_ext;
            }

            // F: vertical gap (gap in query)
            float f_open = H[i-1] - gap_oe;
            float f_ext  = Fval - gap_extend;
            if (f_open > f_ext) {
                Fval = f_open;
                Fs = Hs[i-1];
            } else {
                Fval = f_ext;
            }

            // H: best of match, E, F, or 0
            float h = match_val;
            int hs = match_s;
            if (E[i] > h) { h = E[i]; hs = Es[i]; }
            if (Fval > h) { h = Fval; hs = Fs; }
            if (h < 0.0f) { h = 0.0f; hs = i; }

            h_diag = h_left;
            h_diag_s = h_left_s;
            H[i] = h;
            Hs[i] = hs;

            if (h > max_score) {
                max_score = h;
                best_qi = i;
                best_qj = j;
                best_qstart = hs;
            }
        }
    }

    int out_idx = pair_idx * 5;
    results[out_idx + 0] = max_score;
    results[out_idx + 1] = (float)(best_qstart + 1);  // 1-based qstart
    results[out_idx + 2] = (float)best_qi;
    results[out_idx + 3] = (float)qlen;
    results[out_idx + 4] = (float)tlen;
}
'''

# Simpler CPU fallback using numpy for validation
def _sw_cpu(query, target, blosum62, gap_open=11, gap_extend=1):
    """CPU Smith-Waterman for validation. Returns (score, qstart, qend, qlen, tlen)."""
    m, n = len(query), len(target)
    H = np.zeros((m + 1, n + 1), dtype=np.float32)
    E = np.zeros((m + 1, n + 1), dtype=np.float32)
    F = np.zeros((m + 1, n + 1), dtype=np.float32)

    max_score = 0.0
    max_i, max_j = 0, 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            qi = query[i - 1]
            tj = target[j - 1]

            match = H[i-1, j-1] + blosum62[qi, tj]

            E[i, j] = max(H[i-1, j] - (gap_open + gap_extend),
                          E[i-1, j] - gap_extend)
            F[i, j] = max(H[i, j-1] - (gap_open + gap_extend),
                          F[i, j-1] - gap_extend)

            H[i, j] = max(0, match, E[i, j], F[i, j])

            if H[i, j] > max_score:
                max_score = H[i, j]
                max_i, max_j = i, j

    # Traceback for start position
    i, j = max_i, max_j
    while i > 0 and j > 0 and H[i, j] > 0:
        qi = query[i - 1]
        tj = target[j - 1]
        diag = H[i-1, j-1] + blosum62[qi, tj]
        if H[i, j] == diag and diag > 0:
            i -= 1
            j -= 1
        elif H[i, j] == E[i, j]:
            i -= 1
        else:
            j -= 1

    q_start = i + 1  # 1-based
    q_end = max_i     # 1-based

    return max_score, q_start, q_end, m, n


# Vectorized encoding lookup table
_AA_LOOKUP = np.full(256, AA_TO_IDX['X'], dtype=np.int32)
for _aa, _idx in AA_TO_IDX.items():
    _AA_LOOKUP[ord(_aa)] = _idx


def encode_sequence(seq_str):
    """Convert amino acid string to integer array using BLOSUM62 ordering."""
    return _AA_LOOKUP[np.frombuffer(seq_str.encode('ascii'), dtype=np.uint8)]


class GPUAligner:
    """GPU-accelerated Smith-Waterman aligner for protein sequences.

    Drop-in replacement for BLAST in chewBBACA's allele calling pipeline.
    Uses BLOSUM62 with gap_open=11, gap_extend=1 (BLAST defaults).
    """

    def __init__(self, gap_open=GAP_OPEN, gap_extend=GAP_EXTEND, device=0):
        if cp is None:
            raise ImportError("CuPy is required for GPU alignment. "
                              "Install with: pip install cupy-cuda12x")
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.device = device

        with cp.cuda.Device(self.device):
            self.blosum62_gpu = cp.asarray(BLOSUM62, dtype=cp.int32)
            self._kernel = cp.RawKernel(_SW_KERNEL_CODE, 'smith_waterman_batch')
            # Allow up to 99KB shared memory per block (L4 CC 8.9 supports this)
            # Default is 48KB; this allows ~50% more concurrent blocks per SM
            try:
                self._kernel.max_dynamic_shared_size_bytes = 99 * 1024
            except Exception:
                pass  # fall back to default 48KB

    def align_pairs(self, query_seqs, target_seqs, query_ids, target_ids):
        """Align pairs of sequences on GPU.

        Parameters
        ----------
        query_seqs : list of str
            Query protein sequences.
        target_seqs : list of str
            Target protein sequences (same length as query_seqs).
        query_ids : list of str
            Query sequence identifiers.
        target_ids : list of str
            Target sequence identifiers.

        Returns
        -------
        results : list of tuples
            Each tuple: (qseqid, qstart, qend, qlen, sseqid, slen, score)
            Same format as BLAST outfmt '6 qseqid qstart qend qlen sseqid slen score'
        """
        num_pairs = len(query_seqs)
        if num_pairs == 0:
            return []

        # Encode all sequences
        encoded_queries = [encode_sequence(q) for q in query_seqs]
        encoded_targets = [encode_sequence(t) for t in target_seqs]

        # Compute offsets
        q_offsets = np.zeros(num_pairs, dtype=np.int32)
        t_offsets = np.zeros(num_pairs, dtype=np.int32)
        q_lengths = np.array([len(q) for q in encoded_queries], dtype=np.int32)
        t_lengths = np.array([len(t) for t in encoded_targets], dtype=np.int32)

        q_total = 0
        t_total = 0
        for i in range(num_pairs):
            q_offsets[i] = q_total
            t_offsets[i] = t_total
            q_total += len(encoded_queries[i])
            t_total += len(encoded_targets[i])

        # Flatten sequences
        all_queries = np.concatenate(encoded_queries) if num_pairs > 0 else np.array([], dtype=np.int32)
        all_targets = np.concatenate(encoded_targets) if num_pairs > 0 else np.array([], dtype=np.int32)

        with cp.cuda.Device(self.device):
            d_queries = cp.asarray(all_queries)
            d_targets = cp.asarray(all_targets)
            d_q_offsets = cp.asarray(q_offsets)
            d_t_offsets = cp.asarray(t_offsets)
            d_q_lengths = cp.asarray(q_lengths)
            d_t_lengths = cp.asarray(t_lengths)
            d_results = cp.zeros(num_pairs * 5, dtype=cp.float32)

            # Bucket pairs by query length for tight shared memory allocation
            BUCKET_LIMITS = [128, 256, 512, 1024, 100000]
            q_lens_np = q_lengths  # already numpy

            prev = 0
            for limit in BUCKET_LIMITS:
                mask = (q_lens_np >= prev) & (q_lens_np < limit)
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    prev = limit
                    continue

                bucket_max_qlen = int(q_lens_np[indices].max())
                bucket_max_tlen = int(t_lengths[indices].max())
                n_bucket = len(indices)
                # 4 arrays: H, E (float) + Hs, Es (int) + BLOSUM62 (576 ints)
                shared_mem_bytes = 4 * (bucket_max_qlen + 1) * 4 + 576 * 4

                d_indices = cp.asarray(indices.astype(np.int32))
                sub_results = cp.zeros(n_bucket * 5, dtype=cp.float32)

                self._kernel(
                    (n_bucket,), (1,),
                    (d_queries, d_targets,
                     d_q_offsets[d_indices], d_t_offsets[d_indices],
                     d_q_lengths[d_indices], d_t_lengths[d_indices],
                     self.blosum62_gpu,
                     np.int32(self.gap_open), np.int32(self.gap_extend),
                     sub_results, np.int32(n_bucket),
                     np.int32(bucket_max_qlen), np.int32(bucket_max_tlen)),
                    shared_mem=shared_mem_bytes
                )

                d_results.reshape(num_pairs, 5)[d_indices] = sub_results.reshape(n_bucket, 5)
                prev = limit

            results_np = cp.asnumpy(d_results).reshape(num_pairs, 5)

        # Format as BLAST output
        blast_results = []
        for i in range(num_pairs):
            score = results_np[i, 0]
            if score > 0:
                qstart = int(results_np[i, 1])
                qend = int(results_np[i, 2])
                qlen = int(results_np[i, 3])
                slen = int(results_np[i, 4])
                blast_results.append(
                    (query_ids[i], str(qstart), str(qend), str(qlen),
                     target_ids[i], str(slen), str(score))
                )

        return blast_results

    def align_pairs_raw(self, all_queries, all_targets,
                         q_offsets, t_offsets, q_lengths, t_lengths,
                         query_ids, target_ids):
        """Align pairs using pre-encoded, pre-flattened sequence arrays.

        This avoids redundant encoding when the same sequence appears
        in multiple pairs (e.g., one query vs many targets).

        Splits pairs into buckets by query length to maximize GPU occupancy.
        Shared memory = 3*(max_qlen+1)*4 bytes per block, so grouping
        similar-length queries allows more concurrent blocks per SM.
        """
        num_pairs = len(query_ids)
        if num_pairs == 0:
            return []

        with cp.cuda.Device(self.device):
            d_queries = cp.asarray(all_queries)
            d_targets = cp.asarray(all_targets)
            d_q_offsets = cp.asarray(q_offsets)
            d_t_offsets = cp.asarray(t_offsets)
            d_q_lengths = cp.asarray(q_lengths)
            d_t_lengths = cp.asarray(t_lengths)
            d_results = cp.zeros(num_pairs * 5, dtype=cp.float32)

            # Split pairs into buckets by query length for tight shared memory.
            BUCKET_LIMITS = [128, 256, 512, 1024, 100000]
            q_lens_np = np.asarray(q_lengths)

            prev = 0
            for limit in BUCKET_LIMITS:
                mask = (q_lens_np >= prev) & (q_lens_np < limit)
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    prev = limit
                    continue

                bucket_max_qlen = int(q_lens_np[indices].max())
                bucket_max_tlen = int(np.asarray(t_lengths)[indices].max())
                n_bucket = len(indices)
                # 4 arrays: H, E (float) + Hs, Es (int) + BLOSUM62 (576 ints)
                shared_mem_bytes = 4 * (bucket_max_qlen + 1) * 4 + 576 * 4

                d_indices = cp.asarray(indices.astype(np.int32))
                sub_q_off = d_q_offsets[d_indices]
                sub_t_off = d_t_offsets[d_indices]
                sub_q_len = d_q_lengths[d_indices]
                sub_t_len = d_t_lengths[d_indices]
                sub_results = cp.zeros(n_bucket * 5, dtype=cp.float32)

                self._kernel(
                    (n_bucket,), (1,),
                    (d_queries, d_targets, sub_q_off, sub_t_off,
                     sub_q_len, sub_t_len, self.blosum62_gpu,
                     np.int32(self.gap_open), np.int32(self.gap_extend),
                     sub_results, np.int32(n_bucket),
                     np.int32(bucket_max_qlen), np.int32(bucket_max_tlen)),
                    shared_mem=shared_mem_bytes
                )

                d_results.reshape(num_pairs, 5)[d_indices] = sub_results.reshape(n_bucket, 5)
                prev = limit

            results_np = cp.asnumpy(d_results).reshape(num_pairs, 5)

        blast_results = []
        for i in range(num_pairs):
            score = results_np[i, 0]
            if score > 0:
                qstart = int(results_np[i, 1])
                qend = int(results_np[i, 2])
                qlen = int(results_np[i, 3])
                slen = int(results_np[i, 4])
                blast_results.append(
                    (query_ids[i], str(qstart), str(qend), str(qlen),
                     target_ids[i], str(slen), str(score))
                )

        return blast_results

    def align_all_vs_all(self, sequences, seq_ids):
        """Align all sequences against all others (for cluster BLAST replacement).

        Parameters
        ----------
        sequences : dict
            {seq_id: protein_sequence_string}
        seq_ids : list
            List of sequence IDs to include.

        Returns
        -------
        results : list of tuples
            BLAST-format results for all pairs with score > 0.
        """
        ids = [sid for sid in seq_ids if sid in sequences]
        seqs = [sequences[sid] for sid in ids]
        n = len(ids)

        if n == 0:
            return []

        # Generate all pairs (query, target)
        query_seqs = []
        target_seqs = []
        query_ids = []
        target_ids = []

        for i in range(n):
            for j in range(n):
                query_seqs.append(seqs[i])
                target_seqs.append(seqs[j])
                query_ids.append(ids[i])
                target_ids.append(ids[j])

        return self.align_pairs(query_seqs, target_seqs, query_ids, target_ids)

    def align_cluster(self, query_seqs_dict, target_ids, query_ids=None):
        """Align query sequences against target sequences in a cluster.

        This replaces the per-cluster BLAST call. Each query is aligned
        against each target in the cluster.

        Parameters
        ----------
        query_seqs_dict : dict
            {seq_id: protein_sequence_string} for all sequences.
        target_ids : list
            IDs of target sequences (the cluster members).
        query_ids : list or None
            IDs of query sequences. If None, queries = targets.

        Returns
        -------
        results : list of tuples
            BLAST-format results.
        """
        if query_ids is None:
            query_ids = target_ids

        q_seqs = []
        t_seqs = []
        q_ids = []
        t_ids = []

        for qid in query_ids:
            if qid not in query_seqs_dict:
                continue
            qseq = query_seqs_dict[qid]
            for tid in target_ids:
                if tid not in query_seqs_dict:
                    continue
                tseq = query_seqs_dict[tid]
                q_seqs.append(qseq)
                t_seqs.append(tseq)
                q_ids.append(qid)
                t_ids.append(tid)

        return self.align_pairs(q_seqs, t_seqs, q_ids, t_ids)

    def self_align(self, sequences, seq_ids):
        """Compute self-alignment scores (each sequence against itself).

        Replaces determine_self_scores BLAST calls.

        Parameters
        ----------
        sequences : dict
            {seq_id: protein_sequence_string}
        seq_ids : list
            Sequence IDs to compute self-scores for.

        Returns
        -------
        self_scores : dict
            {seq_id: (dna_length, raw_score)}
            dna_length = (protein_length * 3) + 3 (to count stop codon)
        """
        ids = [sid for sid in seq_ids if sid in sequences]
        seqs = [sequences[sid] for sid in ids]

        results = self.align_pairs(seqs, seqs, ids, ids)

        self_scores = {}
        for r in results:
            qid = r[0]
            qlen = int(r[3])
            score = float(r[6])
            dna_length = (qlen * 3) + 3
            self_scores[qid] = (dna_length, score)

        return self_scores


def align_batch_gpu(pairs, sequences, gap_open=GAP_OPEN, gap_extend=GAP_EXTEND):
    """Convenience function for batch GPU alignment.

    Parameters
    ----------
    pairs : list of tuples
        [(query_id, target_id), ...]
    sequences : dict
        {seq_id: protein_sequence_string}

    Returns
    -------
    results : list of tuples
        BLAST-format results.
    """
    aligner = GPUAligner(gap_open=gap_open, gap_extend=gap_extend)

    q_seqs = [sequences[p[0]] for p in pairs]
    t_seqs = [sequences[p[1]] for p in pairs]
    q_ids = [p[0] for p in pairs]
    t_ids = [p[1] for p in pairs]

    return aligner.align_pairs(q_seqs, t_seqs, q_ids, t_ids)


# Global aligner instance (lazy initialization)
_global_aligner = None

def get_aligner():
    """Get or create the global GPU aligner instance."""
    global _global_aligner
    if _global_aligner is None:
        _global_aligner = GPUAligner()
    return _global_aligner
