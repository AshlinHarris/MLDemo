#!/usr/bin/env julia

#using BenchmarkTools
using DataFrames
#using FilePathsBase
#using JuliaDB
using MLJ
#using Profile
#using Random
#using Revise
#using ScientificTypesBase
#using Traceur

include("src/MLDemo.jl")
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

	# Filter DataFrames
	miscarriage_only = dataframe_subset(conditions_df, "Miscarriage in first trimester")
	with_allergies = dataframe_subset(allergy_df, miscarriage_only)

	# Convert list-style DataFrame to matrix-style DataFrame
	main_df = list_to_matrix(allergy_df)

	# Add target column
	insertcols!(main_df, :MISCARRIAGE => map(Bool, zeros(nrow(main_df))), makeunique = true)
	list = miscarriage_only.PATIENT |> unique
	for x in eachrow(main_df)
		if x[:PATIENT] in list
			x[:MISCARRIAGE] = true
		end
	end

	
	describe(main_df) |> display

	coerce!(main_df, :MISCARRIAGE => OrderedFactor{2})

	(acc, f1_score)= run_decision_tree(main_df, :MISCARRAIGE)
	println()
	println(acc)
	println(f1_score)

	return nothing
end

main()
