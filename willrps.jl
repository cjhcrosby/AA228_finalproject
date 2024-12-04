using Distributions

const ACTIONS = ["rock", "paper", "scissors"]
const NUM_ACTIONS = length(ACTIONS)

# Initialize regrets and strategy sums
regret_sum = zeros(Float64, NUM_ACTIONS)
strategy_sum = zeros(Float64, NUM_ACTIONS)
opponent_strategy = fill(1.0 / NUM_ACTIONS, NUM_ACTIONS)  # Initial opponent strategy is random

# Function to get a normalized strategy based on regrets
function get_strategy(regret_sum)
    strategy = max.(regret_sum, 0)  # Replace negative regrets with zero
    normalizing_sum = sum(strategy)
    if normalizing_sum > 0
        strategy ./= normalizing_sum
    else
        strategy .= 1.0 / NUM_ACTIONS
    end
    return strategy
end

# Function to get a move based on a given strategy
function get_action(strategy)
    return rand(Categorical(strategy))
end

# Function to compute the utility for each action
function get_action_utility(my_action, opponent_action)
    utility = zeros(Float64, NUM_ACTIONS)
    if my_action == 1 && opponent_action == 3 ||  # Rock beats Scissors
       my_action == 2 && opponent_action == 1 ||  # Paper beats Rock
       my_action == 3 && opponent_action == 2     # Scissors beats Paper
        utility[my_action] = 1.0
    elseif my_action == opponent_action
        utility[my_action] = 0.0
    else
        utility[my_action] = -1.0
    end
    return utility
end

# Main CFR training function with self-play
function train_cfr_self_play(iterations)
    global regret_sum, strategy_sum, opponent_strategy
    action_utilities = zeros(Float64, NUM_ACTIONS)
    
    for _ in 1:iterations
        # Get current strategy and sample an action
        strategy = get_strategy(regret_sum)
        my_action = get_action(strategy)
        
        # Opponent plays according to its previous strategy
        opponent_action = get_action(opponent_strategy)
        
        # Calculate utility of actions against the opponent's move
        for a in 1:NUM_ACTIONS
            action_utilities[a] = get_action_utility(a, opponent_action)[a]
        end
        
        # Compute regret
        for a in 1:NUM_ACTIONS
            regret_sum[a] += action_utilities[a] - action_utilities[my_action]
        end
        
        # Update strategy sum for average strategy calculation
        strategy_sum .= strategy_sum .+ strategy
        
        # Update opponent strategy to be the current strategy (self-play)
        opponent_strategy = copy(strategy)
    end
end

# Function to get the average strategy after training
function get_average_strategy()
    normalizing_sum = sum(strategy_sum)
    if normalizing_sum > 0
        return strategy_sum ./ normalizing_sum
    else
        return fill(1.0 / NUM_ACTIONS, NUM_ACTIONS)
    end
end

# Run training
iterations = 10000
train_cfr_self_play(iterations)
average_strategy = get_average_strategy()
println("Average strategy after training (Rock, Paper, Scissors): $average_strategy")
