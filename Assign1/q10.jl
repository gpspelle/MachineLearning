using CSV
using PyPlot
# df = CSV.read("amazon_reviews_us_Gift_Card_v1_00.tsv"; delim='	')
df = CSV.read("small.tsv"; delim='	')
vals = zeros(Int64, 5)

for i = 1:4
	vals[df[8][i]] += 1
end

x = range(1, step=1, length=6)
plot(x, vals, color="red")
