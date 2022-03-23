module MLDemo
export get_data, dataframe_subset, list_to_matrix, top_n_values

using DataFrames

using CSV: File
using StatsBase: countmap

macro nameofvariable(x)
	return string(x)
end

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
	rows = df.PATIENT |> sort |> unique
	cols = df.DESCRIPTION |> sort |> unique

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
	function top_n_values(df, col, n)::Nothing

	Find top n values by occurence
"""
function top_n_values(df, col, n)::Nothing
	name = @nameofvariable(df)
	println("Top $n values for $col in $name:")
	x = first(sort(countmap(df[:, col]; alg = :dict), byvalue = true, rev = true), n)
	show(IOContext(stdout, :limit => false), "text/plain", x)
	println()
	return
end

end