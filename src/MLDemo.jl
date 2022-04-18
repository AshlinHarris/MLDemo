module MLDemo
export add_target_column!, get_data, dataframe_subset, boolean_unstack, number_with, run_decision_tree, top_n_values

using ConfParser # Parse, modify, write to configuration files
using DataFrames
using MLJ
using MLJDecisionTreeInterface

using CSV: File
using StatsBase: countmap

"""
	function add_target_column!(df::AbstractDataFrame, symb::Symbol, target_df::AbstractDataFrame)::Nothing

Add column to a DataFrame based on symbol presence in the target DataFrame 
"""
function add_target_column!(df::AbstractDataFrame, symb::Symbol, target_df::AbstractDataFrame)::Nothing
	#TODO: is it certain the the order of eachrow will match the order of checks?
	insertcols!(df, symb => [x[:PATIENT] in target_df.PATIENT for x in eachrow(df)])
	coerce!(df, symb => OrderedFactor{2})
	return nothing
end


"""
	function get_data(file_name::String)::AbstractDataFrame

Return the contents of a CSV file as a DataFrame
"""
function get_data(file_name::String)::AbstractDataFrame
	conf = ConfParse("./config.ini")
	parse_conf!(conf)
	path = retrieve(conf, "local", "input_directory")
	
	file = joinpath(path, file_name)
	return File(file, header = 1) |> DataFrame
end

function get_directory(file_name::String)::String
	conf = ConfParse("./config.ini")
	parse_conf!(conf)
	path = retrieve(conf, "local", "path")
	
	file = joinpath(path, file_name)
	return File(file, header = 1) |> DataFrame
end


"""
	function dataframe_subset(df::AbstractDataFrame, check::Any)::AbstractDataFrame

Return a DataFrame subset
For check::DataFrame, including only PATIENTs present in check
Otherwise, Subset DataFrame of PATIENTs with condition
"""
function dataframe_subset(df::AbstractDataFrame, check::AbstractDataFrame)::DataFrame
	return filter(:DESCRIPTION => x -> x in check.PATIENT, df)
end
function dataframe_subset(df::AbstractDataFrame, check::Any)::AbstractDataFrame
	return filter(:DESCRIPTION => x -> isequal(x, check), df)
end

"""
	function boolean_unstack(df::AbstractDataFrame, x::Symbol, y::Symbol)::AbstractDataFrame

Unstack a DataFrame df by row and column keys x and y

Isn't there a one-liner for this?
"""
function boolean_unstack(df::AbstractDataFrame, x::Symbol, y::Symbol)::AbstractDataFrame
	B = unstack(combine(groupby(df, [x,y]), nrow => :count), x, y, :count, fill=0)
	for q in names(select(B, Not(:PATIENT)))
		B[!,q] = B[!,q] .!= 0
	end
	return B
end


#TODO: What are the valid types for RNG_VALUE
"""
	function run_decision_tree(df::AbstractDataFrame, output::Symbol, RNG_VALUE)::Tuple{AbstractFloat, AbstractFloat}

Decision tree classifier on a DataFrame over a given output
"""
function run_decision_tree(df::AbstractDataFrame, output::Symbol, RNG_VALUE)::Tuple{AbstractFloat, AbstractFloat}
	y = df[:, output]
	X = select(df, Not([:PATIENT, output]))
	
	train, test = partition(eachindex(y), 0.8, shuffle = true, rng = RNG_VALUE)

	# Evaluate model
	Tree = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
	tree_model = Tree(max_depth = 3)
	evaluate(tree_model, X, y) |> display

	# Return scores
	tree = machine(tree_model, X, y)
	fit!(tree, rows = train)
	yhat = predict(tree, X[test, :])
	acc = accuracy(MLJ.mode.(yhat), y[test])
	f1_score = f1score(MLJ.mode.(yhat), y[test])

	return acc, f1_score
end

"""
	function top_n_values(df::DataFrame, col::Symbol, n::Int)::DataFrame

Find top n values by occurence
"""
function top_n_values(df::DataFrame, col::Symbol, n::Int)::DataFrame
	return first(sort(combine(nrow, groupby(df, col)), "nrow", rev=true), n)
end

end
