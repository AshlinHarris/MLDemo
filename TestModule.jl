module TestModule

export top_n_values

macro nameofvariable(x)
	return string(x)
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