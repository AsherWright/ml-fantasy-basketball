using CSV

COMPETITION_PATH = "../data/competition.csv"
MODEL_INPUT_PATH = "../data/model_input.csv"
TRAINING_DATA_PATH = "../processed/OCTs/oct_train_data.csv"
TREE_PATH = "../processed/OCTs/OCT.json"

# Given a trained tree lnr, and a matrix of , return the mean and std of each node populated by the injected features
function get_nodes_mean_std(lnr, train_X, train_y; only_leafs = false)
    if only_leafs
        leafs = get_leafs(lnr)
        nodes_elements = IAI.apply_nodes(lnr, train_X)[leafs]
    else
        nodes_elements = IAI.apply_nodes(lnr, train_X)
    end
    nb_nodes = length(nodes_elements)
    stds = zeros(length(nodes_elements))
    means = zeros(length(nodes_elements))
    for i=1:nb_nodes
        stds[i] = std(Vector(train_y)[nodes_elements[i]])
        means[i] = mean(Vector(train_y)[nodes_elements[i]])
    end
    means, stds
end;

function get_leafs(lnr)
    num_nodes = IAI.get_num_nodes(lnr)
    leafs = []
    for i=1:num_nodes
        if IAI.is_leaf(lnr, i)
            push!(leafs, i)
        end
    end
    leafs
end;

function get_leaf_pred_mean_std(lnr, new_X, train_X, train_y)
    nodes_means, nodes_stds = get_nodes_mean_std(lnr, train_X, train_y);
    predictions = IAI.predict(lnr, new_X)
    assigned_leafs = IAI.apply(lnr, new_X)
    leafs_means = nodes_means[assigned_leafs]
    leafs_stds = nodes_stds[assigned_leafs]
    DataFrame(assigned_leaf = assigned_leafs,
        prediciton = predictions,
        leaf_mean = leafs_means,
        leaf_std = leafs_stds
    )
end;

function
