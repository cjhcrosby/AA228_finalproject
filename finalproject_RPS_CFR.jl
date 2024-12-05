#= 
AA 228 Final Project 
Colton Crosby, William Ho, Kevin Murillo
=#

using Random
using Distributions
using Plots

# Constants for the game
const NUM_ACTIONS = 3  # Rock, Paper, Scissors
const ACTIONS = 1:NUM_ACTIONS  # Actions labeled as 1 (Rock), 2 (Paper), 3 (Scissors)

# Payoff matrix: actionUtility[i, j] gives the payoff for `i` against `j`
const actionUtility = [
    0 -1  1; # rvr(0), rvp(-1), rvs(1)
    1  0 -1; # pvr(1), pvp(0), pvs(-1)
   -1  1  0  # svr(-1), svp(1), svs(0)
]

# get the strat based on regret
function get_strategy(regret_sum)
    # Clamp regrets to zero
    positive_regrets = max.(regret_sum, 0) # positive regrets are clipped to zero (no negative regrets)
    normalizing_sum = sum(positive_regrets) # normalizing constant
    
    # If there's no positive regret, return a uniform strategy
    if normalizing_sum > 0
        return positive_regrets / normalizing_sum
    else
        return fill(1.0 / NUM_ACTIONS, NUM_ACTIONS)
    end
end

# action from strat prob
function get_action(strategy)
    return rand(Categorical(strategy))  # Sample action based on strategy probabilities
end

function get_average_strategy(strategy_sum)
    normalizing_sum = sum(strategy_sum)
    if normalizing_sum > 0
        return strategy_sum / normalizing_sum
    else
        return fill(1.0 / NUM_ACTIONS, NUM_ACTIONS)
    end
end



function train_cfr_policies_over_time(iterations)
    regret_sum = zeros(NUM_ACTIONS)
    strategy_sum = zeros(NUM_ACTIONS)
    
    opponent_regret_sum = zeros(NUM_ACTIONS)
    opponent_strategy_sum = zeros(NUM_ACTIONS)
    
    # Initialize opponent’s strategy to a uniform distribution
    opponent_strategy = fill(1.0 / NUM_ACTIONS, NUM_ACTIONS)
    
    # Track average policies over time for plotting
    player_policies_over_time = []
    opponent_policies_over_time = []

    for _ in 1:iterations
        # Get current player strategy based on regret sums
        strategy = get_strategy(regret_sum)
        opponent_strategy_copy = [1,0,0]#copy(opponent_strategy)
        
        # Accumulate strategies to calculate average policy later
        strategy_sum += strategy
        opponent_strategy_sum += opponent_strategy_copy
        
        # Store average policies for plotting
        push!(player_policies_over_time, copy(get_average_strategy(strategy_sum)))
        push!(opponent_policies_over_time, copy(get_average_strategy(opponent_strategy_sum)))
        
        # Sample actions for both players
        my_action = get_action(strategy)
        opponent_action = get_action(opponent_strategy_copy)
        
        # Calculate rewards
        my_reward = actionUtility[my_action, opponent_action]
        opp_reward = actionUtility[opponent_action, my_action]
        
        # Update regrets
        for a in ACTIONS
            regret_sum[a] += actionUtility[a, opponent_action] - my_reward
            opponent_regret_sum[a] += actionUtility[a, my_action] - opp_reward
        end
        
        # Update opponent’s strategy to be the player's previous strategy
        opponent_strategy = strategy
    end
    
    return player_policies_over_time, opponent_policies_over_time
end

# Training and storing policies over time
iterations = 100000
player_policies, opponent_policies = train_cfr_policies_over_time(iterations)

# Extract policies for plotting
player_rock = [policy[1] for policy in player_policies]
player_paper = [policy[2] for policy in player_policies]
player_scissors = [policy[3] for policy in player_policies]
# println([player_rock, player_paper, player_scissors])

opponent_rock = [policy[1] for policy in opponent_policies]
opponent_paper = [policy[2] for policy in opponent_policies]
opponent_scissors = [policy[3] for policy in opponent_policies]
# println([opponent_rock, opponent_paper, opponent_scissors])
# Plot player policies over time
plot(1:iterations, player_rock, label="Player Rock", xlabel="Iteration", ylabel="Probability", title="Player Policies Over Time",
)
plot!(1:iterations, player_paper, label="Player Paper")
plot!(1:iterations, player_scissors, label="Player Scissors")

# Plot opponent policies over time
plot!(1:iterations, opponent_rock, label="Opponent Rock", linestyle=:dash)
plot!(1:iterations, opponent_paper, label="Opponent Paper", linestyle=:dash)
plot!(1:iterations, opponent_scissors, label="Opponent Scissors", linestyle=:dash)