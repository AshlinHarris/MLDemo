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
	topic_2_df = get_data("allergies.csv")
	demographics_df = get_data("patients.csv")

	# Filter DataFrames
	topic_1 = "Hypertension"
	#topic_1 = "Miscarriage in first trimester"
	topic_1_only = dataframe_subset(conditions_df, topic_1)
	with_topic_2 = dataframe_subset(topic_2_df, topic_1_only)

	### ALLERGY STUDY ###

	println()
	println("Example study: Allergy associations")
	println()

	# Summarize DataFrames
#=
	for df in [conditions_df, topic_2_df]
		x = top_n_values(df, :DESCRIPTION, 12)
		println(x)
		display(x[!,:DESCRIPTION])
		println()
	end
=#

	# Generate composite DataFrame
	composite_df = boolean_unstack(topic_2_df, :PATIENT, :DESCRIPTION)
	add_target_column!(composite_df, :MISCARRIAGE, topic_1_only)

	if nrow(with_topic_2) != 0
		# Machine learning
		RNG_VALUE = abs(rand(Int))
		acc, f1_score= run_decision_tree(composite_df, :MISCARRIAGE, RNG_VALUE)
		
		# Results
		println()
		@printf("Accuracy: %.3f\n", acc)
		@printf("F1 Score: %.3f\n", f1_score)
	end

	### DEMOGRAPHICS ###

	# Study feasibility
	#selected = number_with(:DESCRIPTION, topic_1, conditions_df)
	selected = nrow(topic_1_only)
	total = nrow(conditions_df)

	println()
	@printf("%s: %s\n", "Study feasibility", topic_1)
	@printf("%30s: %7d\n", "Total number of entries", total)
	@printf("    %26s: %7d (%6.2f%%)\n", "Selected entries", selected, 100 * (selected / total))
	#@printf("%30s: %7d\n", "Total number of entries", nrow(topic_1_only))
	println()
	println()

	# From the demographics DataFrame, take only PATIENTS with "Miscarriage in first trimester"
	#TODO: dataframe_subset() should be generalized to handle this
	topic_1_demographics = filter(:Id => x -> x in topic_1_only.PATIENT, demographics_df)
	#display(topic_1_demographics)
	#display(names(topic_1_demographics))

	println()
	println("Comparison of Demographic information")
	println()
	plots=[]
	FACTORS = [:RACE, :ETHNICITY, :GENDER]
	DATAFRAMES = [demographics_df, topic_1_demographics]
	for factor in FACTORS
		for df in DATAFRAMES
			x = top_n_values(df, factor, 12)
			y = pie(x[!,factor], x.nrow) # Cannot handle missing values
			push!(plots, y)
		end
		a = top_n_values(topic_1_demographics, factor, 12)
		rename!(a, Dict(:nrow => :Miscarriage))
		b = top_n_values(demographics_df, factor, 12)
		rename!(b, Dict(:nrow => :All))
		println(outerjoin(a, b, on=factor, matchmissing=:equal))
		println()
	end
	fig1 = plot(plots..., layout = (length(FACTORS), length(DATAFRAMES)), plot_title="Demographics: All vs Patients with Condition")
	png(fig1, "demographics.png")

	return nothing
end

main()
