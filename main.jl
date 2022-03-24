#!/usr/bin/env julia

#using BenchmarkTools
using DataFrames
#using FilePathsBase
#using JuliaDB
#using Random
#using Revise
#using ScientificTypesBase

using MLJ
#load_path("DecisionTreeClassifier", pkg="DecisionTree")
using MLJDecisionTreeInterface

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
	@time top_n_values(conditions_df, :DESCRIPTION, 12)
	@time top_n_values(allergy_df, :DESCRIPTION, 12)

	# Filter DataFrames
	miscarriage_only = dataframe_subset(conditions_df, "Miscarriage in first trimester")
	with_allergies = dataframe_subset(allergy_df, miscarriage_only)

	# Convert list-style DataFrame to matrix-style DataFrame
	A = list_to_matrix(allergy_df)

	# Add target column
	insertcols!(A, :MISCARRIAGE => map(Bool, zeros(nrow(A))), makeunique = true)
	list = miscarriage_only.PATIENT |> unique
	for x in eachrow(A)
		if x[:PATIENT] in list
			x[:MISCARRIAGE] = true
		end
	end

	describe(A) |> display

	df = A
	coerce!(df, :MISCARRIAGE => OrderedFactor{2})

	y = df.MISCARRIAGE
	X = select(df, Not([:PATIENT, :MISCARRIAGE]))

	RNG_VALUE = 2022
	train, test = partition(eachindex(y), 0.8, shuffle = true, rng = RNG_VALUE)
	#display(models(matching(X, y)))
	println()

	Tree = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
	tree_model = Tree(max_depth = 3)
	tree = machine(tree_model, X, y)

	fit!(tree, rows = train)
	yhat = predict(tree, X[test, :])

	acc = accuracy(MLJ.mode.(yhat), y[test]) |> display
	println()
	f1_score = f1score(MLJ.mode.(yhat), y[test]) |> display

	return nothing
end

@time main()
