#!/usr/bin/env julia

using DataFrames
using MLJ
using Printf
using Random
using Revise

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

	# Study feasibility
	topic = "Miscarriage in first trimester"
	selected = number_with(:DESCRIPTION, topic, conditions_df)
	total = nrow(conditions_df)

#=
	title_format = "%30s: %7d"
	header_format = "%30s: %7d"
	line_format = "    %26s: %7d (%6.3f)"
=#

	@printf("%s: %s\n", "Study feasibility", topic)
	@printf("%30s: %7d\n", "Total number of entries", total)
	@printf("    %26s: %7d (%6.2f%%)\n", "Selected entries", selected, 100 * (selected / total))

	# Summarize DataFrames
	println(top_n_values(conditions_df, :DESCRIPTION, 12))
	println()
	println(top_n_values(allergies_df, :DESCRIPTION, 12))
	println()

	# Filter DataFrames
	miscarriage_only = dataframe_subset(conditions_df, topic)
	#@printf("%30s: %7d\n", "Total number of entries", nrow(miscarriage_only))
	with_allergies = dataframe_subset(allergies_df, miscarriage_only)

	# Generate composite DataFrame
	composite_df = boolean_unstack(allergies_df, :PATIENT, :DESCRIPTION)
	add_target_column!(composite_df, :MISCARRIAGE, miscarriage_only)

	# Machine learning
	RNG_VALUE = abs(rand(Int))
	acc, f1_score= run_decision_tree(composite_df, :MISCARRIAGE, RNG_VALUE)
	
	# Results
	println()
	@printf("Accuracy: %.3f\n", acc)
	@printf("F1 Score: %.3f\n", f1_score)

	return nothing
end

main()
