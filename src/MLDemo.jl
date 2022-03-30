__precompile__()
module MLDemo
export add_target_column!, get_data, dataframe_subset, boolean_unstack, run_decision_tree, top_n_values

using ConfParser # Parse, modify, write to configuration files
using DataFrames
using MLJ
#load_path("DecisionTreeClassifier", pkg="DecisionTree")
using MLJDecisionTreeInterface

using CSV: File
using StatsBase: countmap

macro nameofvariable(x)
	return string(x)
end

"""
	function add_target_column!(df, symb, target_df)

Add column to a DataFrame based on symbol presence in the target DataFrame 
"""
function add_target_column!(df::DataFrame, symb, target_df::DataFrame)
	insertcols!(df, symb => map(Bool, zeros(nrow(df))), makeunique = true)
	list = target_df.PATIENT |> unique
	for x in eachrow(df)
		if x[:PATIENT] in list
			x[symb] = true
		end
	end
	#coerce!(df, symb => OrderedFactor{2}) # Why doesn't this work here?
end

"""
	function get_data(file_name)::DataFrame

Return the contents of a CSV file as a DataFrame
"""
function get_data(file_name)::DataFrame

	conf = ConfParse("./config.ini")
	parse_conf!(conf)
	path = retrieve(conf, "local", "path")
	
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
	function boolean_unstack(df::DataFrame)::DataFrame

Unstack a DataFrame df by row and column keys x and y

Isn't there a one-liner for this?
"""
function boolean_unstack(df::DataFrame, x::Symbol, y::Symbol)::DataFrame

	#=
	###OLD METHOD###
	rows = df[!,x] |> sort |> unique
	cols = df[!,y] |> sort |> unique

	r_dict = Dict()
	for k in 1:length(rows)
		r_dict[rows[k]] = k
	end
	c_dict = Dict()
	for k in 1:length(cols)
		c_dict[cols[k]] = k
	end

	A = zeros(length(rows), length(cols))
	for q in eachrow(df)
		i = r_dict[q[x]]
		j = c_dict[q[y]]
		A[i, j] = true
	end

	A = DataFrame([Vector{Bool}(undef, length(rows)) for _ in eachcol(A)], :auto)
	rename!(A, cols)
	insertcols!(A, 1, x => rows, makeunique = true)
	=#

	###NEW METHOD###
	B = unstack(combine(groupby(df, [x,y]), nrow => :count), x, y, :count, fill=0)
	for q in names(select(B, Not(:PATIENT)))
		B[!,q] = B[!,q] .!= 0
	end
	
	###EXPERIMENTAL###
	#B = DataFrame(colwise(col -> recode(col, missing=>0), B), names(B)
	#B = combine(groupby(df, [x,y]), nrow => :count)

	return B
end

"""
	function run_decision_tree(df, output)

Decision tree classifier on a DataFrame over a given output
"""
function run_decision_tree(df::DataFrame, output)
	output = :MISCARRIAGE

	y = df[:, output]
	X = select(df, Not([:PATIENT, output]))
	
	#TODO: make sure this is deterministic
	RNG_VALUE = 2022
	train, test = partition(eachindex(y), 0.8, shuffle = true, rng = RNG_VALUE)
	#display(models(matching(X, y)))
	println()

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
	function top_n_values(df, col, n)::Nothing

Find top n values by occurence
"""
function top_n_values(df::DataFrame, col, n)::Nothing
	name = @nameofvariable(df)
	println("Top $n values for $col in $name:")
	#=
	x = first(sort(countmap(df[:, col]; alg = :dict), byvalue = true, rev = true), n)
	show(IOContext(stdout, :limit => false), "text/plain", x)
	println()
	=#
	first(combine(nrow, groupby(df, col)), n) |> println
	return
end

end
