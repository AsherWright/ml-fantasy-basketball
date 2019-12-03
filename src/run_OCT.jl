using Pkg;
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Statistics")
using CSV, DataFrames, Statistics;

df = CSV.read("../data/all_games_7_players.csv")

train_proportion = 0.6
validation_proportion = 0.2

X = df[:, 15:406]
y = df[:, end]
(train_X, train_y), (test_valid_X, test_valid_y) = IAI.split_data(:regression, X, y, seed=1, train_proportion=0.6);
(valid_X, valid_y), (test_X, test_y) = IAI.split_data(:regression, test_valid_X, test_valid_y, seed=1, train_proportion=validation_proportion/(1-train_proportion));

# Validation parameters
max_depth=5:15
minbucket=[10]

# Default learner
default_lnr = IAI.OptimalTreeRegressor(
    random_seed=1,
    criterion=:mse,
    );

# Grid
grid = IAI.GridSearch(default_lnr,
    max_depth=max_depth,
    cp=cp,
    minbucket=minbucket
);

# Fitting the grid
IAI.fit!(grid, train_X, train_y, valid_X, valid_y);

# Retrieving best parameters
best_params = IAI.get_best_params(grid)

# Retriving the best tree
lnr = IAI.get_learner(grid)

# Getting the 1 - misclassification error
train_accuracy = IAI.score(lnr,train_X, train_y, criterion=:mse)
valid_accuracy = IAI.score(lnr,valid_X, valid_y, criterion=:mse)
test_accuracy = IAI.score(lnr,test_X, test_y, criterion=:mse)

lnr = IAI.write_json("../processed/OCTs/all_players.json");
