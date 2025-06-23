#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------



Code documentation
------------------
"""


import os
import sys

import pandas as pd

try:
	from GetAlleles import get_alleles
	from utils import (
		constants as ct,
		mafft_wrapper as mw,
		file_operations as fo,
		fasta_operations as fao,
		iterables_manipulation as im,
		multiprocessing_operations as mo)
except ModuleNotFoundError:
	from CHEWBBACA.GetAlleles import get_alleles
	from CHEWBBACA.utils import (
		constants as ct,
		mafft_wrapper as mw,
		file_operations as fo,
		fasta_operations as fao,
		iterables_manipulation as im,
		multiprocessing_operations as mo)


def concatenate_loci_alignments(sample, loci, sample_profile, fasta_index, output_directory):
	"""Concatenate the aligned sequences for a sample.

	Parameters
	----------
	sample : str
		Sample identifier.
	loci : list
		Loci identifiers.
	fasta_index : Bio.File._IndexedSeqFileDict
		Indexed FASTA file to get sequences from.
	output_directory : str
		Path to the output directory.

	Returns
	-------
	alignment_outfile : str
		Path to the FASTA file with the concatenated aligned sequences.
	"""
	alignment = ''
	for locus in loci:
		# Sequence headers include sample, locus, and allele IDs joined by '_'
		# Get allele ID
		allele_id = sample_profile[locus].tolist()[0]
		# Get aligned sequence from index
		try:
			seqid = f'{sample}_{locus}_{allele_id}'
			alignment += str(fasta_index[seqid].seq)
		except Exception as e:
			seqid = f'{sample}_{locus}_0'
			alignment += str(fasta_index[seqid].seq)
	# Save alignment for sample
	alignment_outfile = fo.join_paths(output_directory,
									  [f'{sample}_cgMLST_alignment.fasta'])
	alignment_record = fao.fasta_str_record(ct.FASTA_RECORD_TEMPLATE,
											[sample, alignment])
	fo.write_lines([alignment_record], alignment_outfile)

	return alignment_outfile


def add_gaps(input_file, locus_id, gap_char, sample_ids, output_directory):
	"""Add gap sequences to a MSA for samples that do not have an allele.

	Parameters
	----------
	input_file : str
		Path to the FASTA file with the aligned sequences.
	locus_id : str
		Locus identifier.
	gap_char : str
		Character to use to fill gaps.
	sample_ids : list
		Sample identifiers.
	output_directory : str
		Path to the output directory.

	Returns
	-------
	gapped_fasta : str
		Path to the FASTA file containing the updated MSA.
	"""
	records = fao.import_sequences(input_file)
	# Get list of samples where locus was identified
	dataset_sids = {k.split(f'_{locus_id}')[0]: k for k in records}
	# Get length of alignment to create gapped sequence
	# Just get length of the first record
	msa_len = len(records[list(records.keys())[0]])
	gapped_seq = gap_char * msa_len
	gapped_records = {}
	for sid in sample_ids:
		# Sample contains locus
		if sid in dataset_sids:
			gapped_records[dataset_sids[sid]] = records[dataset_sids[sid]]
		# Sample does not contain locus
		# Add gapped sequence
		else:
			gapped_seq_sid = f'{sid}_{locus_id}_0'
			gapped_records[gapped_seq_sid] = gapped_seq

	# Save Fasta file with gapped sequences
	gapped_fasta = fo.join_paths(output_directory, [f'{locus_id}_gapped.fasta'])
	outrecords = [f'>{k}\n{v}' for k, v in gapped_records.items()]
	fo.write_lines(outrecords, gapped_fasta)

	return gapped_fasta


def convert_msa_to_dna(input_file, dna_file, locus_id, gap_char, output_directory):
	"""
	"""
	protein_records = fao.import_sequences(input_file)
	dna_sequences = fao.import_sequences(dna_file)
	dna_records = {}
	for seqid, sequence in protein_records.items():
		# Check if it matches any record in the schema
		if seqid in dna_sequences:
			# Get allele sequence
			allele = dna_sequences[seqid]
			# Iterate over gapped protein sequence to create gapped DNA
			dna_index = 0
			gapped_dna = ''
			for i, char in enumerate(sequence):
				# Add codon if it is not a gap
				if char != gap_char:
					gapped_dna += allele[dna_index:dna_index+3]
					dna_index += 3
				# Add '---' if it is a gap
				elif char == gap_char:
					gapped_dna += gap_char * 3
			dna_records[seqid] = gapped_dna
		else:
			dna_records[seqid] = sequence * 3

	# Save DNA MSA
	dna_fasta = fo.join_paths(output_directory, [f'{locus_id}_gapped_dna.fasta'])
	dna_msa_recs = [f'>{k}\n{v}' for k, v in dna_records.items()]
	fo.write_lines(dna_msa_recs, dna_fasta)

	return dna_fasta

# Test
input_file = '/home/rmamede/test_chewie/features/ComputeMSA/cdiff_enterobase_data/20250422_Enterobase_cgMLST_filtered_dated_only_lc099_si099_flt_samples_matrix_100perc.tsv'
schema_directory = '/home/rmamede/test_chewie/features/ComputeMSA/cdiff_enterobase_data/Cdiff_cgMLST_schema_Enterobase'
output_directory = '/home/rmamede/test_chewie/features/ComputeMSA/cdiff_enterobase_data/test_msa'
dna_msa = True
output_variable = True
gap_char = '-'
translation_table = 11
cpu_cores = 12
keep_locus_msa = False
only_locus_msa = False
def main(input_file, schema_directory, output_directory, dna_msa, output_variable, gap_char, translation_table, cpu_cores, keep_locus_msa, only_locus_msa):
	# Create output directory
	fo.create_directory(output_directory)
	# Get sample and loci IDs
	sample_ids = fo.extract_column(input_file, delimiter='\t', column_index=0)
	loci_ids = fo.read_lines(input_file, strip=True, num_lines=1)[0].split('\t')[1:]

	# Create FASTA files with the alleles identified in the dataset
	# Call GetAlleles module
	_, dna_files, protein_files = get_alleles.main(input_file, schema_directory, output_directory, cpu_cores, False, True, translation_table)

	# Get FASTA files for loci identified in at least one sample
	if len(dna_files) == 0:
		sys.exit(ct.COMPUTEMSA_NO_ALLELES)

	# Run MAFFT to compute MSA
	print('\nRunning MAFFT to compute the MSA for each locus...')
	mafft_outdir = fo.join_paths(output_directory, ['mafft_results'])
	fo.create_directory(mafft_outdir)
	mafft_outfiles = [os.path.basename(file) for file in protein_files]
	mafft_outfiles = [file.replace('.fasta', '_aligned.fasta') for file in mafft_outfiles]
	mafft_outfiles = [fo.join_paths(mafft_outdir, [file]) for file in mafft_outfiles]
	mafft_inputs = [[file, mafft_outfiles[i]] for i, file in enumerate(protein_files)]
	common_args = []
	# Add common arguments to all sublists
	inputs = im.multiprocessing_inputs(mafft_inputs, common_args, mw.call_mafft)
	mafft_results = mo.map_async_parallelizer(inputs,
											  mo.function_helper,
											  cpu_cores,
											  show_progress=True)

	# Identify cases where MAFFT failed
	mafft_failed = [r[0] for r in mafft_results if r[1] is False]
	if len(mafft_failed) > 0:
		print(f'\nCould not determine MSA for {len(mafft_failed)} loci.')

	# Get files created by MAFFT
	mafft_success = [r[0] for r in mafft_results if r[1] is True]

	# Add gap sequences when sample did not have an allele
	gapped_inputs = []
	for file in mafft_success:
		locus_id = fo.file_basename(file, False).split('_protein')[0]
		gapped_inputs.append([file, locus_id])

	common_args = [gap_char, sample_ids, mafft_outdir]
	gapped_inputs = im.multiprocessing_inputs(gapped_inputs, common_args, add_gaps)
	gapped_results = mo.map_async_parallelizer(gapped_inputs,
											   mo.function_helper,
											   cpu_cores,
											   show_progress=True)

	# Delete original file with ungapped MSA
	fo.remove_files(mafft_success)

	# Convert protein MSAs to DNA MSAs
	if dna_msa:
		dna_inputs = []
		for i, file in enumerate(gapped_results):
			locus_id = fo.file_basename(file).split('_gapped')[0]
			dna_file = dna_files[i]
			dna_inputs.append([file, dna_file, locus_id])

		common_args = [gap_char, mafft_outdir]
		dna_inputs = im.multiprocessing_inputs(dna_inputs, common_args, convert_msa_to_dna)
		dna_results = mo.map_async_parallelizer(dna_inputs,
										        mo.function_helper,
												cpu_cores,
												show_progress=True)

	# Delete FASTA files with DNA and protein sequences
	fo.delete_directory(os.path.dirname(dna_files[0]))
	fo.delete_directory(os.path.dirname(protein_files[0]))

##############################
# Create function to get variable positions
# Need to concatenate SNP MSAs to get complete SNP MSA
# Should exclude/ignore N's and gaps when determining variable positions for DNA MSA?
# Need to save output files for protein, DNA and variable MSAs into separate folders

	# Identify variable positions to get SNP MSA
	if output_variable:
		# Determine variable positions for protein MSAs
		for file in gapped_results:
			locus_id = fo.file_basename(file).split('_gapped')[0]
			# Read MSA
			msa_records = fao.import_sequences(file)
			msa_sequences = list(msa_records.values())
			# Zip to pair chars in same positions
			zipped = list(zip(*msa_sequences))
			variable = []
			for i, pos in enumerate(zipped):
				# Get unique characters in position
				distinct = set(pos)
				# If there are more than 1 unique character, it is a variable position
				if len(distinct) > 1:
					variable.append([i, pos])
			if len(variable) > 0:
				variable_pos = list(zip(*[i[1] for i in variable]))
				variable_pos = [''.join(i) for i in variable_pos]
				variable_records = [[allele_id, variable_pos[i]] for i, allele_id in enumerate(msa_records.keys())]
				variable_records = fao.fasta_lines(ct.FASTA_RECORD_TEMPLATE, variable_records)
				output_file = fo.join_paths(output_directory, [locus_id + '_variable.fasta'])
				fo.write_lines(variable_records, output_file)
				# Save variable positions
			else:
				print(f'Locus {locus_id} has no variable positions. Will not write SNP MSA.')

##############################

	# User only wants the locus MSAs
	# Do not compute full MSAs
	if only_locus_msa:
		return

	# Create folder to store sample MSAs
	sample_msa_folder = fo.join_paths(output_directory, ['sample_alignments'])
	fo.create_directory(sample_msa_folder)

	# Create the full protein MSA
	print('\nCreating file with the full protein MSA...')
	# Concatenate all alignment files and index with BioPython
	protein_concat = fo.join_paths(sample_msa_folder, [ct.COMPUTEMSA_PROTEIN_CONCAT])
	fo.concatenate_files(gapped_results, protein_concat)
	# Index file
	indexed_protein_concat = fao.index_fasta(protein_concat)
	sample_alignment_files = []
	# Get loci IDs
	loci_ids = [fo.file_basename(file, False).split('_gapped')[0] for file in gapped_results]
	msa_length = 0
	for sid in sample_ids:
		# Get sample profile
		sample_profile = pd.read_csv(input_file,
								     skiprows=range(1,sample_ids.index(sid)+1), # Only get header and sample profile
									 nrows=1,
									 delimiter='\t',
									 header=0,
									 dtype=str)
		alignment_file = concatenate_loci_alignments(sid,
													 loci_ids,
													 sample_profile,
													 indexed_protein_concat,
													 sample_msa_folder)
		sample_alignment_files.append(alignment_file)

	# Concatenate sample protein alignments
	full_alignment = fo.join_paths(output_directory, [ct.COMPUTEMSA_PROTEIN_MSA])
	fo.concatenate_files(sample_alignment_files, full_alignment)

	# Get length of the full alignment
	msa_length = len(fo.read_lines(full_alignment, strip=True, num_lines=2)[1])
	print(f'Protein MSA length: {msa_length}')

	if dna_msa:
		# Create the full DNA MSA
		print('Creating file with the full DNA MSA...')
		# Concatenate all alignment files and index with BioPython
		dna_concat = fo.join_paths(sample_msa_folder, [ct.COMPUTEMSA_DNA_CONCAT])
		fo.concatenate_files(dna_results, dna_concat)
		# Index file
		indexed_dna_concat = fao.index_fasta(dna_concat)
		sample_alignment_files = []
		# Get loci IDs
		loci_ids = [fo.file_basename(file, False).split('_gapped')[0] for file in dna_results]
		for sid in sample_ids:
			# Get sample profile
			sample_profile = pd.read_csv(input_file,
										 skiprows=range(1,sample_ids.index(sid)+1),
										 nrows=1,
										 delimiter='\t',
										 header=0,
										 dtype=str)
			alignment_file = concatenate_loci_alignments(sid,
														 loci_ids,
														 sample_profile,
														 indexed_dna_concat,
														 sample_msa_folder)
			sample_alignment_files.append(alignment_file)

		# Concatenate all cgMLST alignmnet records
		full_alignment = fo.join_paths(output_directory, [ct.COMPUTEMSA_DNA_MSA])
		fo.concatenate_files(sample_alignment_files, full_alignment)

		# Get length of the full alignment
		msa_length = len(fo.read_lines(full_alignment, strip=True, num_lines=2)[1])
		print(f'Protein MSA length: {msa_length}')

	if not keep_locus_msa:
		fo.delete_directory(mafft_outdir)

	# Delete folder with sample MSAs
	fo.delete_directory(sample_msa_folder)
