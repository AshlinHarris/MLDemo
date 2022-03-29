#!/usr/bin/env julia

#using BenchmarkTools
using DataFrames
#using Dates
#using Distributed
#using FilePathsBase
#using JuliaDB
#using LinearAlgebra
using MLJ
#using Profile
#using Random
using Revise
#using ScientificTypesBase
#using SparseArrays
#using Statistics
#using Traceur

includet("src/MLDemo.jl")
using .MLDemo

"""
	function main()

Machine Learning in Julia with Synthetic Data
"""
function main()

	# Read in DataFrames from files
	conditions_df = get_data("conditions.csv")
	allergy_df = get_data("allergies.csv")

	# Summarize DataFrames
	
	top_n_values(conditions_df, :DESCRIPTION, 12)
	top_n_values(allergy_df, :DESCRIPTION, 12)

#	describe(conditions_df) |> println

	# Filter DataFrames
	miscarriage_only = dataframe_subset(conditions_df, "Miscarriage in first trimester")
	with_allergies = dataframe_subset(allergy_df, miscarriage_only)

	# Generate composite DataFrame
	composite_df = boolean_unstack(allergy_df, :PATIENT, :DESCRIPTION)
	#composite_df |> describe |> println
	#composite_df = pivotView(allergy_df, :PATIENT, :DESCRIPTION)
	#pivotView(allergy_df, :PATIENT, :DESCRIPTION) |> describe |> println
	add_target_column!(composite_df, :MISCARRIAGE, miscarriage_only)
	coerce!(composite_df, :MISCARRIAGE => OrderedFactor{2})
#	describe(composite_df) |> display

	# Machine learning
	(acc, f1_score)= run_decision_tree(composite_df, :MISCARRAIGE)
	
	# Results
	println()
	println(acc)
	println(f1_score)

	return nothing
end

main()
