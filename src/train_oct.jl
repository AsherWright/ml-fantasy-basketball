using CSV, DataFrames, Statistics;

TRAINING_DATA_PATH = "../data/all_games_7_players.csv"

# Reading data
historic_data = CSV.read(TRAINING_DATA_PATH)
X = historic_data[:, 2:end-1]
y = historic_data[:, :fp_p1]

# Selecting training columns
all_columns = names(X)
exclude_players_cols = all_columns
# exclude_players_cols = all_columns[.!endswith.(string.(all_columns),["p6"])]
# exclude_players_cols = exclude_players_cols[.!endswith.(string.(exclude_players_cols),["p7"])]
# exclude_players_cols = exclude_players_cols[.!endswith.(string.(exclude_players_cols),["p13"])]
# exclude_players_cols = exclude_players_cols[.!endswith.(string.(exclude_players_cols),["p14"])]
p1_cols = exclude_players_cols[endswith.(string.(all_columns), "_p1")];
fp_avg_cols =  exclude_players_cols[occursin.("fp_seas_avg", string.(all_columns))];
fp_l_cols =  exclude_players_cols[occursin.("fp_l5", string.(all_columns))];
train_cols = unique(vcat(p1_cols, fp_avg_cols, fp_l_cols));
X = X[:,train_cols];

train_proportion = 0.6
validation_proportion = 0.2
(train_X, train_y), (test_valid_X, test_valid_y) = IAI.split_data(:regression, X, y, seed=1, train_proportion=train_proportion);
(valid_X, valid_y), (test_X, test_y) = IAI.split_data(:regression, test_valid_X, test_valid_y, seed=1, train_proportion=validation_proportion/(1-train_proportion));

# Validation parameters
MAX_DEPTH = 7:10
CP=[0.0001, 0.00001]

# Default learner
default_lnr = IAI.OptimalTreeRegressor(
    random_seed=1,
    criterion=:mse,
    minbucket=10
    );

# Grid
grid = IAI.GridSearch(default_lnr,
    max_depth=MAX_DEPTH,
    cp=CP
);

print(grid)

# Fitting the grid
IAI.fit!(grid, train_X, train_y, valid_X, valid_y);

# Getting the best get_learner
lnr = IAI.get_learner(grid);

# Retrieving best parameters
best_params = IAI.get_best_params(grid)
println(best_params)

grid_results = IAI.get_grid_results(grid)
println(grid_results)

var_importance = IAI.variable_importance(lnr)
println(var_importance)

train_accuracy = IAI.score(lnr,train_X, train_y, criterion=:mse);
valid_accuracy = IAI.score(lnr,valid_X, valid_y, criterion=:mse);
test_accuracy = IAI.score(lnr,test_X, test_y, criterion=:mse);
train_MAE = mean(abs.(IAI.predict(lnr, train_X) - train_y));
valid_MAE = mean(abs.(IAI.predict(lnr, valid_X) - valid_y));
test_MAE = mean(abs.(IAI.predict(lnr, test_X) - test_y));

println(string("Train R2 : ", train_accuracy))
println(string("Train MAE : ", train_MAE))

println(string("Valid R2 : ", valid_accuracy))
println(string("Valid MAE : ", valid_MAE))

println(string("Test R2 : ", test_accuracy))
println(string("Test MAE : ", test_MAE))

IAI.write_html("../processed/OCTs/OCT.html", lnr);
IAI.write_json("../processed/OCTs/OCT.json", lnr);

train_X[:, :fp_p1] = convert(Array,train_y)
valid_X[:, :fp_p1] = convert(Array,valid_y)
test_X[:, :fp_p1] = convert(Array,test_y)

CSV.write("../processed/OCTs/oct_train_data.csv", train_X)
CSV.write("../processed/OCTs/oct_valid_data.csv", valid_X)
CSV.write("../processed/OCTs/oct_test_data.csv", test_X)
