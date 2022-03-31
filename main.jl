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
using Random
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

	SEED_VALUE = 2022
	Random.seed!(SEED_VALUE)

	# Read in DataFrames from files
	conditions_df = get_data("conditions.csv")
	allergies_df = get_data("allergies.csv")

	# Summarize DataFrames
	top_n_values(conditions_df, :DESCRIPTION, 12) |> println
	top_n_values(allergies_df, :DESCRIPTION, 12) |> println

	# Filter DataFrames
	miscarriage_only = dataframe_subset(conditions_df, "Miscarriage in first trimester")
	with_allergies = dataframe_subset(allergies_df, miscarriage_only)

	# Generate composite DataFrame
	composite_df = boolean_unstack(allergies_df, :PATIENT, :DESCRIPTION)
	add_target_column!(composite_df, :MISCARRIAGE, miscarriage_only)

	# Machine learning
	RNG_VALUE = abs(rand(Int))
	acc, f1_score= run_decision_tree(composite_df, :MISCARRIAGE, RNG_VALUE)
	
	# Results
	println()
	println("Accuracy: ", round(acc, digits=4))
	println("F1 Score: ", round(f1_score, digits=4))

	return nothing
end

main()
