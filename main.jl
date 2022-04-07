#!/usr/bin/env julia

using DataFrames
using MLJ
using Plots
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

	### ALLERGY STUDY ###

	println()
	println("Example study: allergy associations")
	println()

	# Generate composite DataFrame
	composite_df = boolean_unstack(allergies_df, :PATIENT, :DESCRIPTION)
	#display(composite_df)
	add_target_column!(composite_df, :MISCARRIAGE, miscarriage_only)

	# Machine learning
	#TODO: Skip this if total==0
	RNG_VALUE = abs(rand(Int))
	acc, f1_score= run_decision_tree(composite_df, :MISCARRIAGE, RNG_VALUE)
	
	# Results
	println()
	@printf("Accuracy: %.3f\n", acc)
	@printf("F1 Score: %.3f\n", f1_score)

	### DEMOGRAPHICS ###

	# Study feasibility
	#selected = number_with(:DESCRIPTION, topic, conditions_df)
	selected = nrow(miscarriage_only)
	total = nrow(conditions_df)

	println()
	@printf("%s: %s\n", "Study feasibility", topic)
	@printf("%30s: %7d\n", "Total number of entries", total)
	@printf("    %26s: %7d (%6.2f%%)\n", "Selected entries", selected, 100 * (selected / total))
	#@printf("%30s: %7d\n", "Total number of entries", nrow(miscarriage_only))
	println()
	println()

	# From the demographics DataFrame, take only PATIENTS with "Miscarriage in first trimester"
	#TODO: dataframe_subset() should be generalized to handle this
	miscarriage_demographics = filter(:Id => x -> x in miscarriage_only.PATIENT, demographics_df)
	#display(miscarriage_demographics)
	#display(names(miscarriage_demographics))

	println()
	println("Comparison of Demographic information")
	println()
	for category in [:MARITAL, :RACE, :ETHNICITY, :GENDER]
		a = top_n_values(miscarriage_demographics, category, 12)
		rename!(a, Dict(:nrow => :Miscarriage))
		b = top_n_values(demographics_df, category, 12)
		rename!(b, Dict(:nrow => :All))
		println(outerjoin(a, b, on=category, matchmissing=:equal))
	println()
	end

	#=
	#missing values an issue?
		df = top_n_values(demographics_df, :MARITAL, 12)
		a1 = df.MARITAL, df.nrow
		df = top_n_values(miscarriage_demographics, :MARITAL, 12)
		a2 = df.MARITAL, df.nrow
	=#

	plots=[]
	for df in [demographics_df, miscarriage_demographics]
		for factor in [:RACE, :ETHNICITY, :GENDER]
			x = top_n_values(df, factor, 12)
			y = pie(x[!,factor], x.nrow)
			push!(plots, y)
		end
	end
	fig1 = plot(plots..., layout = (3,2))
	png(fig1, "demographics.png")

	return nothing
end

main()
