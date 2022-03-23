#!/usr/bin/env julia

#using BenchmarkTools
using DataFrames
#using FilePathsBase
#using JuliaDB
using MLJ
#using Random
#using ScientificTypesBase

using CSV: File
using StatsBase: countmap

#include("TestModule.jl")
using .TestModule

"""
	function get_data(file_name)::DataFrame

Return the contents of a CSV file as a DataFrame
"""
function get_data(file_name)::DataFrame

	fp = open("config.txt")
	path= readline(fp)
	close(fp)

	file = joinpath(path, file_name)
	return File(file, header = 1) |> DataFrame
end

"""
	function dataframe_subset(df::DataFrame, check::Any)::DataFrame

Return a DataFrame subset
For check::DataFrame, including only PATIENTs present in check
Otherwise, Subset DataFrame of PATIENTs with condition
"""
function dataframe_subset(df::DataFrame, check::DataFrame)::DataFrame
	return filter(:DESCRIPTION => x -> x in check.PATIENT, df)
end
function dataframe_subset(df::DataFrame, check::Any)::DataFrame
	return filter(:DESCRIPTION => x -> isequal(x, check), df)
end

"""
	function list_to_matrix(df::DataFrame)::DataFrame

Convert list-style DataFrame to matrix-style DataFrame
"""
function list_to_matrix(df::DataFrame)::DataFrame
	rows = df.PATIENT |> sort |> unique # |> enumerate
	cols = df.DESCRIPTION |> sort |> unique # |> enumerate

	r_dict = Dict()
	for k in 1:length(rows)
		r_dict[rows[k]] = k
	end
	c_dict = Dict()
	for k in 1:length(cols)
		c_dict[cols[k]] = k
	end

	A = zeros(length(rows), length(cols))
	for x in eachrow(df)
		i = r_dict[x.PATIENT]
		j = c_dict[x.DESCRIPTION]
		A[i, j] = true
	end

	A = DataFrame([Vector{Bool}(undef, length(rows)) for _ in eachcol(A)], :auto)
	rename!(A, cols)
	insertcols!(A, 1, :PATIENT => rows, makeunique = true)

	return A
end

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
	display(models(matching(X, y)))

	Tree = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
	tree_model = Tree(max_depth = 3)
	tree = machine(tree_model, X, y)

	fit!(tree, rows = train)
	yhat = predict(tree, X[test, :])

	acc = accuracy(MLJ.mode.(yhat), y[test]) |> display
	f1_score = f1score(MLJ.mode.(yhat), y[test]) |> display

	return nothing
end

main()
