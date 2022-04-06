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
	demographics_df = get_data("patients.csv")

	# Summarize DataFrames
	println(top_n_values(conditions_df, :DESCRIPTION, 12))
	println()
	println(top_n_values(allergies_df, :DESCRIPTION, 12))
	println()

	# Filter DataFrames
	topic = "Miscarriage in first trimester"
	miscarriage_only = dataframe_subset(conditions_df, topic)
	with_allergies = dataframe_subset(allergies_df, miscarriage_only)

	# Study feasibility

	#selected = number_with(:DESCRIPTION, topic, conditions_df)
	selected = nrow(miscarriage_only)
	total = nrow(conditions_df)

	@printf("%s: %s\n", "Study feasibility", topic)
	@printf("%30s: %7d\n", "Total number of entries", total)
	@printf("    %26s: %7d (%6.2f%%)\n", "Selected entries", selected, 100 * (selected / total))
	#@printf("%30s: %7d\n", "Total number of entries", nrow(miscarriage_only))

	### ALLERGY STUDY ###

	# Generate composite DataFrame
	composite_df = boolean_unstack(allergies_df, :PATIENT, :DESCRIPTION)
	#display(composite_df)
	add_target_column!(composite_df, :MISCARRIAGE, miscarriage_only)

	# Machine learning
	RNG_VALUE = abs(rand(Int))
	acc, f1_score= run_decision_tree(composite_df, :MISCARRIAGE, RNG_VALUE)
	
	# Results
	println()
	@printf("Accuracy: %.3f\n", acc)
	@printf("F1 Score: %.3f\n", f1_score)

	### DEMOGRAPHICS ###
	
	# From the demographics DataFrame, take only PATIENTS with "Miscarriage in first trimester"
	#TODO: dataframe_subset() should be generalized to handle this
	miscarriage_demographics = filter(:Id => x -> x in miscarriage_only.PATIENT, demographics_df)
	#display(miscarriage_demographics)


	return nothing
end

main()
