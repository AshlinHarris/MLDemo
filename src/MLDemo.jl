module MLDemo_old
#export add_target_column!, dataframe_subset, boolean_unstack, number_with, run_decision_tree
export dataframe_subset

using DataFrames
using MLJ
using MLJDecisionTreeInterface

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
	function dataframe_subset(df::AbstractDataFrame, check::Any)::AbstractDataFrame

Return a DataFrame subset
For check::DataFrame, including only PATIENTs present in check
Otherwise, Subset DataFrame of PATIENTs with condition
Condition column name is given by symb
"""
function dataframe_subset(df::AbstractDataFrame, check::AbstractDataFrame, symb::Symbol)::DataFrame
	return filter(symb => x -> x in check.PATIENT, df)
end
function dataframe_subset(df::AbstractDataFrame, check::Any, symb::Symbol)::AbstractDataFrame
	return filter(symb => x -> isequal(x, check), df)
end

"""
	function boolean_unstack(df::AbstractDataFrame, x::Symbol, y::Symbol)::AbstractDataFrame

Unstack a DataFrame df by row and column keys x and y

Isn't there a one-liner for this?
"""
function boolean_unstack(df::AbstractDataFrame, x::Symbol, y::Symbol)::AbstractDataFrame
	B = unstack(combine(groupby(df, [x,y]), nrow => :count), x, y, :count, fill=0)
	for q in names(select(B, Not(x)))
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

end
