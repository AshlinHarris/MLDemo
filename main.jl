#!/usr/bin/env julia

using ColorSchemes
using DataFrames
using MLJ
using Plots
using Printf
using Random
using Revise
using StatsPlots

includet("src/MLDemo.jl")
using .MLDemo

function clean()
	foreach(rm, filter(endswith(".png"), readdir()))
end


"""
	function main()

Machine Learning in Julia with Synthetic Data
"""
function main()

	SEED_VALUE = 2022
	Random.seed!(SEED_VALUE)

	# Read in DataFrames from files
	conditions_df = get_data("conditions.csv")
	demographics_df = get_data("patients.csv")

	# Generate topics
	TOPICS=[]
	TOPICS = top_n_values(conditions_df, :DESCRIPTION, 2)[!,:DESCRIPTION]
	#push!(TOPICS, "Miscarriage in first trimester")
	push!(TOPICS, "Anemia (disorder)")

	# Demographic DataFrames
	DEMOGRAPHICS = Dict()
	FACTORS = [:RACE, :ETHNICITY, :GENDER]
	for factor in FACTORS
		df = top_n_values(demographics_df, factor, 12)
		rename!(df, Dict(:nrow => :Total))
		push!(DEMOGRAPHICS, factor => df)
	end

	# Filter DataFrames
	for i in 1:length(TOPICS)
		topic_1 = TOPICS[i]

		topic_1_only = dataframe_subset(conditions_df, topic_1)

		### ALLERGY STUDY ###

		SCREEN_WIDTH = 60
		println("="^SCREEN_WIDTH)
		#println("Example study: Allergy associations")

		topic_2_df = get_data("allergies.csv")
		with_topic_2 = dataframe_subset(topic_2_df, topic_1_only)

		# Generate composite DataFrame
		composite_df = boolean_unstack(topic_2_df, :PATIENT, :DESCRIPTION)
		add_target_column!(composite_df, :MISCARRIAGE, topic_1_only)

		if nrow(with_topic_2) != 0
			# Machine learning
			RNG_VALUE = abs(rand(Int))
			acc, f1_score= run_decision_tree(composite_df, :MISCARRIAGE, RNG_VALUE)
			
			# Results
			@printf("Accuracy: %.3f\n", acc)
			@printf("F1 Score: %.3f\n", f1_score)
		end

		### DEMOGRAPHICS ###

		# Study feasibility
		selected = nrow(topic_1_only)
		total = nrow(conditions_df)

		@printf("%s: %s\n", "Study feasibility", topic_1)
		@printf("%30s: %7d\n", "Total number of entries", total)
		@printf("    %26s: %7d (%6.2f%%)\n", "Selected entries", selected, 100 * (selected / total))
		#@printf("%30s: %7d\n", "Total number of entries", nrow(topic_1_only))

		# From the demographics DataFrame, take only PATIENTS with the primary topic condition
		#TODO: dataframe_subset() should be generalized to handle this
		topic_1_demographics = filter(:Id => x -> x in topic_1_only.PATIENT, demographics_df)
		#display(topic_1_demographics)
		#display(names(topic_1_demographics))

		#println("Comparison of Demographic information:")
		plots=[]
		DATAFRAMES = [demographics_df, topic_1_demographics]
		for factor in FACTORS
			for df in DATAFRAMES
				x = top_n_values(df, factor, 12)
				y = pie(x[!,factor], x.nrow, color_palette = palette(:Paired_8)) # Cannot handle missing values
				plot!(fontfamily="Computer Modern")
				plot!(label="f")
				push!(plots, y)
			end
		a = top_n_values(topic_1_demographics, factor, 12)
		rename!(a, Dict(:nrow => topic_1))

		DEMOGRAPHICS[factor] = outerjoin(DEMOGRAPHICS[factor], a, on=factor, matchmissing=:equal)

		end
	fig1 = plot(plots..., layout = (length(FACTORS), length(DATAFRAMES)))
	plot!(plot_title="$topic_1")
	savefig(fig1, "demographics_$i.png")

	end

	i=1
	for factor in FACTORS
		df = DEMOGRAPHICS[factor]

		ctg = repeat(df[!,factor], outer = ncol(df)-1)
		nam = repeat(names(df[:, Not(factor)]), inner = nrow(df))
		display(Matrix(df[:,Not(factor)]))

		fig = groupedbar(nam, Matrix(df[:, Not(factor)]), group = ctg)
		plot!(xlabel = "Groups", ylabel = "Individuals")
		plot!(fig, title = "$factor")
		plot!(fig, bar_width = 0.67, lw = 0, framestyle = :box)

		savefig(fig, "bars_$i")
		i=i+1
	end


	return nothing



end

main()
