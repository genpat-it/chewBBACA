/*
 * kmer_filter.c - Fast 6-mer pre-filter for protein sequence pair selection.
 *
 * Build:
 *   gcc -O3 -march=native -shared -fPIC -o kmer_filter.so kmer_filter.c
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define KMER_SIZE 6
#define MIN_SHARED 2

/* FNV-1a hash for 6-byte k-mer */
static inline uint32_t hash_kmer(const char *s) {
    uint32_t h = 0x811c9dc5u;
    for (int i = 0; i < KMER_SIZE; i++) {
        h ^= (uint8_t)s[i];
        h *= 0x01000193u;
    }
    return h;
}

/* ---- Open-addressing hash table: kmer -> (offset, count) into flat array ---- */

typedef struct {
    uint32_t *hashes;
    char     *kmers;      /* KMER_SIZE bytes per entry */
    int32_t  *offsets;    /* -1 = empty */
    int32_t  *counts;
    int32_t   capacity;
    int32_t   mask;
} KmerHT;

static int ht_init(KmerHT *ht, int n_entries) {
    int cap = 16;
    while (cap < n_entries * 2) cap <<= 1;
    ht->capacity = cap;
    ht->mask = cap - 1;
    ht->hashes  = (uint32_t *)calloc(cap, sizeof(uint32_t));
    ht->kmers   = (char *)calloc(cap, KMER_SIZE);
    ht->offsets = (int32_t *)malloc(cap * sizeof(int32_t));
    ht->counts  = (int32_t *)calloc(cap, sizeof(int32_t));
    if (!ht->hashes || !ht->kmers || !ht->offsets || !ht->counts) return -1;
    memset(ht->offsets, -1, cap * sizeof(int32_t));
    return 0;
}

static void ht_free(KmerHT *ht) {
    free(ht->hashes); free(ht->kmers); free(ht->offsets); free(ht->counts);
}

static int32_t ht_find_or_insert(KmerHT *ht, const char *kmer, uint32_t h) {
    int32_t idx = h & ht->mask;
    while (1) {
        if (ht->offsets[idx] == -1) {
            ht->hashes[idx] = h;
            memcpy(ht->kmers + (int64_t)idx * KMER_SIZE, kmer, KMER_SIZE);
            ht->offsets[idx] = -2;  /* occupied, no offset yet */
            ht->counts[idx] = 0;
            return idx;
        }
        if (ht->hashes[idx] == h &&
            memcmp(ht->kmers + (int64_t)idx * KMER_SIZE, kmer, KMER_SIZE) == 0)
            return idx;
        idx = (idx + 1) & ht->mask;
    }
}

static int32_t ht_lookup(const KmerHT *ht, const char *kmer, uint32_t h) {
    int32_t idx = h & ht->mask;
    while (1) {
        if (ht->offsets[idx] == -1) return -1;
        if (ht->hashes[idx] == h &&
            memcmp(ht->kmers + (int64_t)idx * KMER_SIZE, kmer, KMER_SIZE) == 0)
            return idx;
        idx = (idx + 1) & ht->mask;
    }
}

/* ---- Per-sequence k-mer seen set (for deduplication within a sequence) ---- */

typedef struct {
    uint32_t *hashes;    /* 0 = empty */
    char     *kmers;
    int32_t   capacity;
    int32_t   mask;
} SeenSet;

static void seen_init(SeenSet *s, int min_cap) {
    int cap = 64;
    while (cap < min_cap * 2) cap <<= 1;
    s->capacity = cap;
    s->mask = cap - 1;
    s->hashes = (uint32_t *)calloc(cap, sizeof(uint32_t));
    s->kmers  = (char *)calloc(cap, KMER_SIZE);
}

static void seen_clear(SeenSet *s) {
    memset(s->hashes, 0, s->capacity * sizeof(uint32_t));
}

static void seen_free(SeenSet *s) {
    free(s->hashes); free(s->kmers);
}

/* Returns 1 if already seen, 0 if newly inserted */
static int seen_test_insert(SeenSet *s, const char *kmer, uint32_t h) {
    uint32_t h2 = h | 1;  /* ensure non-zero */
    int32_t idx = h2 & s->mask;
    while (1) {
        if (s->hashes[idx] == 0) {
            s->hashes[idx] = h2;
            memcpy(s->kmers + (int64_t)idx * KMER_SIZE, kmer, KMER_SIZE);
            return 0;
        }
        if (s->hashes[idx] == h2 &&
            memcmp(s->kmers + (int64_t)idx * KMER_SIZE, kmer, KMER_SIZE) == 0)
            return 1;
        idx = (idx + 1) & s->mask;
    }
}

/* ---- Main filter ---- */

int kmer_filter_pairs(
    const char *q_seqs, const int *q_offsets, int n_queries,
    const char *t_seqs, const int *t_offsets, int n_targets,
    int *out_qidx, int *out_tidx, int max_pairs)
{
    /* --- Phase 1: Count unique k-mers per target (deduplicated) --- */

    /* Estimate unique k-mers across all targets */
    int total_positions = 0;
    for (int ti = 0; ti < n_targets; ti++) {
        int tlen = t_offsets[ti + 1] - t_offsets[ti];
        if (tlen >= KMER_SIZE) total_positions += tlen - KMER_SIZE + 1;
    }

    /* Build hash table */
    KmerHT ht;
    int est = total_positions;
    if (est < 1024) est = 1024;
    if (ht_init(&ht, est) < 0) return -1;

    /* Seen set for per-target k-mer deduplication */
    /* Size for longest sequence */
    int max_tlen = 0;
    for (int ti = 0; ti < n_targets; ti++) {
        int tlen = t_offsets[ti + 1] - t_offsets[ti];
        if (tlen > max_tlen) max_tlen = tlen;
    }
    SeenSet tseen;
    seen_init(&tseen, max_tlen > 0 ? max_tlen : 64);

    /* Pass 1: Count unique targets per k-mer (each target counted once) */
    for (int ti = 0; ti < n_targets; ti++) {
        int tstart = t_offsets[ti];
        int tlen = t_offsets[ti + 1] - tstart;
        if (tlen < KMER_SIZE) continue;
        const char *tseq = t_seqs + tstart;

        seen_clear(&tseen);
        for (int i = 0; i <= tlen - KMER_SIZE; i++) {
            const char *kmer = tseq + i;
            uint32_t h = hash_kmer(kmer);
            /* Only count this k-mer once per target */
            if (seen_test_insert(&tseen, kmer, h)) continue;
            int32_t slot = ht_find_or_insert(&ht, kmer, h);
            ht.counts[slot]++;
        }
    }

    /* Compute CSR offsets and allocate flat targets array */
    int total_entries = 0;
    for (int i = 0; i < ht.capacity; i++) {
        if (ht.offsets[i] != -1) {
            int32_t cnt = ht.counts[i];
            ht.offsets[i] = total_entries;
            total_entries += cnt;
            ht.counts[i] = 0;  /* reset for pass 2 */
        }
    }

    int32_t *targets = (int32_t *)malloc((size_t)total_entries * sizeof(int32_t));
    if (!targets) { ht_free(&ht); seen_free(&tseen); return -1; }

    /* Pass 2: Fill targets array (deduplicated, no sorting needed) */
    for (int ti = 0; ti < n_targets; ti++) {
        int tstart = t_offsets[ti];
        int tlen = t_offsets[ti + 1] - tstart;
        if (tlen < KMER_SIZE) continue;
        const char *tseq = t_seqs + tstart;

        seen_clear(&tseen);
        for (int i = 0; i <= tlen - KMER_SIZE; i++) {
            const char *kmer = tseq + i;
            uint32_t h = hash_kmer(kmer);
            if (seen_test_insert(&tseen, kmer, h)) continue;
            int32_t slot = ht_lookup(&ht, kmer, h);
            if (slot >= 0) {
                int32_t off = ht.offsets[slot] + ht.counts[slot];
                targets[off] = ti;
                ht.counts[slot]++;
            }
        }
    }

    seen_free(&tseen);

    /* --- Phase 2: Filter queries --- */

    int32_t *qcounts = (int32_t *)calloc(n_targets, sizeof(int32_t));
    if (!qcounts) { free(targets); ht_free(&ht); return -1; }

    /* Track touched targets for fast cleanup */
    int32_t *touched = (int32_t *)malloc((size_t)n_targets * sizeof(int32_t));
    if (!touched) { free(qcounts); free(targets); ht_free(&ht); return -1; }

    /* Size query seen set for longest query */
    int max_qlen = 0;
    for (int qi = 0; qi < n_queries; qi++) {
        int qlen = q_offsets[qi + 1] - q_offsets[qi];
        if (qlen > max_qlen) max_qlen = qlen;
    }
    SeenSet qseen;
    seen_init(&qseen, max_qlen > 0 ? max_qlen : 64);

    int n_pairs = 0;

    for (int qi = 0; qi < n_queries; qi++) {
        int qstart = q_offsets[qi];
        int qlen = q_offsets[qi + 1] - qstart;
        if (qlen < KMER_SIZE) continue;
        const char *qseq = q_seqs + qstart;

        int n_touched = 0;
        int n_matching_kmers = 0;

        seen_clear(&qseen);

        for (int i = 0; i <= qlen - KMER_SIZE; i++) {
            const char *kmer = qseq + i;
            uint32_t h = hash_kmer(kmer);
            if (seen_test_insert(&qseen, kmer, h)) continue;

            int32_t slot = ht_lookup(&ht, kmer, h);
            if (slot < 0) continue;

            n_matching_kmers++;
            int32_t off = ht.offsets[slot];
            int32_t cnt = ht.counts[slot];
            for (int j = 0; j < cnt; j++) {
                int32_t tidx = targets[off + j];
                if (qcounts[tidx] == 0)
                    touched[n_touched++] = tidx;
                qcounts[tidx]++;
            }
        }

        if (n_matching_kmers < MIN_SHARED) {
            for (int j = 0; j < n_touched; j++)
                qcounts[touched[j]] = 0;
            continue;
        }

        /* Emit pairs */
        for (int j = 0; j < n_touched; j++) {
            int32_t tidx = touched[j];
            if (qcounts[tidx] >= MIN_SHARED) {
                if (n_pairs >= max_pairs) {
                    seen_free(&qseen); free(touched);
                    free(qcounts); free(targets); ht_free(&ht);
                    return -1;
                }
                out_qidx[n_pairs] = qi;
                out_tidx[n_pairs] = tidx;
                n_pairs++;
            }
            qcounts[tidx] = 0;
        }
    }

    seen_free(&qseen);
    free(touched);
    free(qcounts);
    free(targets);
    ht_free(&ht);
    return n_pairs;
}
