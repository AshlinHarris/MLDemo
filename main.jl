#!/usr/bin/env julia

using ColorSchemes
using ConfParser # Parse, modify, write to configuration files
using CSV: File
using DataFrames: DataFrame
using DataFrames: ncol
using DataFrames: Not
using DataFrames: nrow
using DataFrames: outerjoin
using DataFrames: rename!
using MLJ
using Plots
using Random: Random
using Revise
using PreprocessMD
using StatsPlots: groupedbar

includet("src/MLDemo.jl")
using .MLDemo_old

global OUTFILES = []

"""
	function clean()

Remove outfiles produced by main()
"""
function clean()
	foreach(rm, OUTFILES)
	global OUTFILES = []
end


"""
	function main()

Machine Learning in Julia with Synthea
"""
function main()

	SEED_VALUE = 2022
	Random.seed!(SEED_VALUE)

	global OUTFILES = []

	MY_COLOR_PALETTE = palette(:Paired_8)

	# Read in DataFrames from files
	conf = ConfParse("./config.ini")
	parse_conf!(conf)
	IN_DIR = retrieve(conf, "local", "input_directory")
	if(isempty(IN_DIR))
		@error "MLDEMO: Input directory must be specified in config.ini!"
	end
	OUT_DIR = retrieve(conf, "local", "output_directory")
	if(isempty(OUT_DIR))
		@error "MLDEMO: Input directory must be specified in config.ini!"
	end

	conditions_df   = File(joinpath(IN_DIR, "conditions.csv"), header = 1) |> DataFrame
	demographics_df = File(joinpath(IN_DIR,   "patients.csv"), header = 1) |> DataFrame
	topic_2_df      = File(joinpath(IN_DIR,  "allergies.csv"), header = 1) |> DataFrame

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
		topic_1_only = MLDemo_old.dataframe_subset(conditions_df, topic_1, :DESCRIPTION)
		#with_topic_2 = MLDemo_old.dataframe_subset(topic_2_df, topic_1_only, :DESCRIPTION)
		#with_topic_2 |> display

		# Generate composite DataFrame
		composite_df = pivot(topic_2_df, :PATIENT, :DESCRIPTION)
		add_label_column!(composite_df, topic_1_only, :MISCARRIAGE)

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
		file_name = joinpath(OUT_DIR, "demographics_$i.png")
		savefig(fig1, file_name)
		push!(OUTFILES, file_name)
		

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

		file_name = joinpath(OUT_DIR, "bars_$i.png")
		savefig(fig, file_name)
		push!(OUTFILES, file_name)
	end

	return nothing
end
"""
	omop()

Experimental pipeline for OMOP Common Data Format
"""
function omop()

	SEED_VALUE = 2022
	Random.seed!(SEED_VALUE)

	# Read in DataFrames from files
	conf = ConfParse("./config.ini")
	parse_conf!(conf)
	IN_DIR = "/home/ashlin/GitHub/mimic-iv-demo-data-in-the-omop-common-data-model-0.9/1_omop_data_csv"
	OUT_DIR = retrieve(conf, "local", "output_directory")

	conditions_df   = File(joinpath(IN_DIR, "condition_era.csv"), header = 1) |> DataFrame

	n = 5
	column_name = :condition_concept_id
	conditions_df = top_n_values(conditions_df, column_name, n)
	println(conditions_df)
	TOPICS = conditions_df[!,column_name]
	for i in 1:length(TOPICS)
		topic_1 = TOPICS[i]
		#topic_1_only = dataframe_subset(conditions_df, topic_1, column_name)
	end

	return nothing
end

main()
