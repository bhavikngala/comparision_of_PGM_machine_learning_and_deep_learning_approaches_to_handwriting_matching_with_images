library(bnlearn)

and_features = read.csv(file = './../../data/features.csv', header = TRUE, sep = ',')

graph <- hc(and_features)

plot(graph)