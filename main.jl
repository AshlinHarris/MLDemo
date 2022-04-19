#!/usr/bin/env julia

using ColorSchemes
using DataFrames
using MLJ
using Plots
#using Printf
using Random
using Revise
using StatsPlots

includet("src/MLDemo.jl")
using .MLDemo

#=
# THIS FUNCTION DELETES ALL PNG FILES IN THE DIRECTORY
function clean()
	foreach(rm, filter(endswith(".png"), readdir()))
end
=#


"""
	function main()

Machine Learning in Julia with Synthetic Data
"""
function main()

	SEED_VALUE = 2022
	Random.seed!(SEED_VALUE)

	#MY_COLOR_PALETTE = palette(:Paired_8)
	#MY_COLOR_PALETTE = palette(:bone, 5)
	MY_COLOR_PALETTE = palette(:tab10)
	#MY_COLOR_PALETTE = palette(:linear_bmy_10_95_c71_n256, 5)
	#MY_COLOR_PALETTE = palette(:tol_bright)

	# Read in DataFrames from files
	conditions_df = get_data("conditions.csv")
	demographics_df = get_data("patients.csv")

	n = 12
	println("#########################################")
	println("#  Top $n condition classifiers by count:")
	println("#########################################")
	println( top_n_values(conditions_df, :DESCRIPTION, n))
	println()
	println()

	# Generate topics (Short and full names)
	TOPICS=[
		["Miscarriage", "Miscarriage in first trimester"],
		["Retinopathy", "Diabetic retinopathy associated with type II diabetes mellitus (disorder)"],
		["Obesity", "Body mass index 30+ - obesity (finding)"],
		["Hypertension", "Hypertension"],
		["Prediabetes", "Prediabetes"],
	]

	# Demographic DataFrames
	DEMOGRAPHICS = Dict()
	FACTORS = [:RACE, :ETHNICITY, :GENDER]
	for factor in FACTORS
		df = top_n_values(demographics_df, factor, 12)
		rename!(df, Dict(:nrow => :Total))
		push!(DEMOGRAPHICS, factor => df)
	end

	FEASIBILITY = DataFrame(Set=["Total"], Number=nrow(conditions_df), Percentage=[100.0], Accuracy=[NaN], F1=[NaN])

	println("#########################################")
	println("#  Machine learning:")
	println("#########################################")
	println()

	# Filter DataFrames
	for i in 1:length(TOPICS)
		short_name, topic_1 = TOPICS[i]
		println("+----------------------------------------")
		println("|  $short_name:")
		println("+----------------------------------------")

		# DataFrame subsets
		topic_1_only = dataframe_subset(conditions_df, topic_1)
		topic_2_df = get_data("allergies.csv")
		#with_topic_2 = dataframe_subset(topic_2_df, topic_1_only)
		#with_topic_2 |> display

		topic_2_df |> display

		# Generate composite DataFrame
		composite_df = boolean_unstack(topic_2_df, :PATIENT, :DESCRIPTION)
		add_target_column!(composite_df, :MISCARRIAGE, topic_1_only)

		composite_df |> display

		# Machine learning
		if nrow(composite_df) == 0
			acc, f1_score = NaN, NaN
		else
			RNG_VALUE = abs(rand(Int))
			acc, f1_score= run_decision_tree(composite_df, :MISCARRIAGE, RNG_VALUE)
		end

		# From the demographics DataFrame, take only PATIENTS with the primary topic condition
		#TODO: dataframe_subset() should be generalized to handle this
		topic_1_demographics = filter(:Id => x -> x in topic_1_only.PATIENT, demographics_df)

		plots=[]
		DATAFRAMES = [demographics_df, topic_1_demographics]
		for factor in FACTORS
			# Demographics pie charts
			for df in DATAFRAMES
				x = top_n_values(df, factor, 12)
				y = pie(x[!,factor], x.nrow, color_palette = MY_COLOR_PALETTE) # Cannot handle missing values
				#plot!(fontfamily="Computer Modern")
				#plot!(label="f")
				push!(plots, y)
			end

			# Push demographics statistics to DataFrame
			a = top_n_values(topic_1_demographics, factor, 12)
			rename!(a, Dict(:nrow => short_name))
			DEMOGRAPHICS[factor] = outerjoin(DEMOGRAPHICS[factor], a, on=factor, matchmissing=:equal)
		end

		# Print pie charts for demographics
		fig1 = plot(plots..., layout = (length(FACTORS), length(DATAFRAMES)))
		plot!(plot_title = window_title = "$short_name")
		savefig(fig1, get_outfile("demographics_$i.png"))

		# Study feasibility
		selected = nrow(topic_1_only)
		total = nrow(conditions_df)
		push!(FEASIBILITY, [short_name, selected, 100 * (selected / total), acc, f1_score])

		println()
		println()
	end

	println("#########################################")
	println("#  Aggregate demographic data:")
	println("#########################################")
	sort(FEASIBILITY, :Number, rev=true) |> println


	# Demographics bar charts
	for i in 1:length(FACTORS)
		factor = FACTORS[i]
		df = coalesce.(DEMOGRAPHICS[factor],0)
		println(df)

		ctg = repeat(df[!,factor], outer = ncol(df)-1)
		nam = repeat(names(df[:, Not(factor)]), inner = nrow(df))

		fig = groupedbar(nam, Matrix(df[:, Not(factor)]), group = ctg, color_palette = MY_COLOR_PALETTE)
		plot!(xlabel = "Groups", ylabel = "Individuals")
		plot!(fig, plot_title = window_title = "$factor")
		plot!(legend=:topleft)

		savefig(fig, get_outfile("bars_$i.png"))
	end

	return nothing
end

main()
