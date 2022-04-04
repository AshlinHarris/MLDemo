module MLDemo
export add_target_column!, get_data, dataframe_subset, boolean_unstack, run_decision_tree, top_n_values

using ConfParser # Parse, modify, write to configuration files
using DataFrames
using MLJ
using MLJDecisionTreeInterface

using CSV: File
using StatsBase: countmap

macro nameofvariable(x)
	return string(x)
end


"""
	function add_target_column!(df::AbstractDataFrame, symb::Symbol, target_df::AbstractDataFrame)::Nothing

Add column to a DataFrame based on symbol presence in the target DataFrame 
"""
function add_target_column!(df::AbstractDataFrame, symb::Symbol, target_df::AbstractDataFrame)::Nothing
	### OLD VERSION ###
#=
	insertcols!(df, symb => zeros(Bool, nrow(df)))
	list = target_df.PATIENT |> unique
	for x in eachrow(df)
		x[symb] = x[:PATIENT] in list
	end
=#
	### NEW VERSION ###
	# Can I be certain that the row ordering is the same?
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

	Tree = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
	tree_model = Tree(max_depth = 3)
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
