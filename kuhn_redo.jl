#= 
AA 228 Final Project 
Colton Crosby, William Ho, Kevin Murillo
=#

# code referenced: https://github.com/Erionis/Kuhn_Poker_CFR/blob/main/CFR_Kuhn.ipynb


using Random
using Distributions
using DataStructures
using Plots
using StatsBase
using PrettyTables

const NUM_ACTIONS = 2
const NUM_CARDS   = 3
const NUM_PLAYERS = 2

mutable struct InfoSet
    key 
    regret_sum::Vector{Float64}
    strategy_sum::Vector{Float64}
    strategy::Vector{Float64}
    reach_pr
    reach_pr_sum
    card_val
end

function next_strategy(infoset::InfoSet)
    infoset.strategy_sum += infoset.reach_pr * infoset.strategy
    infoset.strategy = get_strategy(infoset)
    infoset.reach_pr_sum += infoset.reach_pr # add reach probability to sum
    infoset.reach_pr = 0 # reset reach probability
end

function get_strategy(infoset::InfoSet)::Vector{Float64}
    # Compute the strategy using regrets
    positive_regrets = max.(infoset.regret_sum, 0.0)
    normalizing_sum = sum(positive_regrets)

    if normalizing_sum > 0
        strategy = positive_regrets / normalizing_sum
    else
        strategy = fill(1.0 / NUM_ACTIONS, NUM_ACTIONS) # Uniform strategy
    end
    return strategy
end

function get_average_strategy(infoset::InfoSet)::Vector{Float64}
    strategy = infoset.strategy_sum / infoset.reach_pr_sum 
    strategy = ifelse.(strategy .< 0.001, 0, strategy) # remove negligible strategies 
    normalizing_sum = sum(strategy)
    strategy /= normalizing_sum
    return strategy
end

function card2str(card) 
    if card == 1
        return "J" 
    elseif card == 2
        return "Q" 
    elseif card ==3
        return "K"
    else
        error("CARD NOT CODED")
    end
end

function get_infoset(i_map, card, history) 
    key = join([card2str(card), history])
    # infoset = nothing
    
    if haskey(i_map, key) == 0
        init_regret_sum = zeros(NUM_ACTIONS)
        init_strategy_sum = zeros(NUM_ACTIONS)
        numac_vec = ones(NUM_ACTIONS)
        init_strategy = numac_vec/length(numac_vec)
        init_reach_pr = 0
        init_reach_pr_sum = 0
        card_val = card
        infoset = InfoSet(key, init_regret_sum, init_strategy_sum, init_strategy, init_reach_pr, init_reach_pr_sum,card_val)
        i_map[key] = infoset 
        return infoset
    end

    # println(i_map[key])
    return i_map[key]
end

function cfr(i_map, history="", card_1=-1, card_2=-1, pr_1=1, pr_2=1, pr_c=1)
    """
    Counterfactual regret minimization algorithm.

    Parameters
    ----------
    i_map: information set dictionary
    history : [{'r', 'c', 'b'}], string representing the path taken in the game tree
    'r': random chance action
    'c': check action
    'b': bet action
    card_1 : player 1's card
    card_2 : player 2's card
    pr_1 : Probability that player 1 reaches history
    pr_2 : Probability that player 2 reaches history
    pr_c: Probability contribution of the chance node to reach history
    """

    if is_chance_node(history)
        return chance_util(i_map)
    end

    if is_terminal(history)
        return terminal_util(history, card_1, card_2)
    end

    n = length(history) 
    is_player_1 = n%2 ==0 # is player 1 logical
    if is_player_1
        card = card_1
    else
        card = card_2
    end
    # get infoset for current player card
    infoset = get_infoset(i_map, card, history)
    strategy = infoset.strategy

    # reach probability update
    # println("reacH_pr before ", infoset.reach_pr, " sum: ", infoset.reach_pr_sum)
    if is_player_1
        infoset.reach_pr += pr_1 
    else
        infoset.reach_pr += pr_2 
    end
    # println("pr_1 = ", pr_1, ", pr_2 = ", pr_2, ", pr_c = ", pr_c)
    # println("After: ", infoset.key, " ReachProb:  ", infoset.reach_pr, " Summed: ", infoset.reach_pr_sum)
    # initialize counterfactual utilities over actions
    action_utils = zeros(NUM_ACTIONS) 
    
    # recurse CFR over actions
    for (i, action) in enumerate(["c","b"])
        next_history = join([history, action])

        if is_player_1
            action_utils[i] = -1 * cfr(i_map, next_history,
                                        card_1, card_2, 
                                        pr_1*strategy[i],pr_2,pr_c)
        else
            action_utils[i] = -1 * cfr(i_map, next_history,
                                        card_1, card_2, 
                                        pr_1, pr_2*strategy[i], pr_c)
        end
    end

    ev = sum(action_utils .* strategy)
    regrets = action_utils .- ev
    # println("ev: ", ev)
    # println("regrets: ", regrets)

    if is_player_1
        infoset.regret_sum += pr_2 * pr_c * regrets
    else
        infoset.regret_sum += pr_1 * pr_c * regrets
    end
    # println("regret_sum: ", infoset.regret_sum)
    # println("ev: ", ev)
    return ev
end

function is_chance_node(history)
    return (history == "") # history is empty if chance node
end

function chance_util(i_map)
    ev = 0
    n_possible = NUM_CARDS * NUM_PLAYERS
    for i in 1:NUM_CARDS
        for j in 1:NUM_CARDS
            if i != j 
                ev += cfr(i_map, "  ", i, j, 1, 1, 1/n_possible)
            end
        end
    end
    return ev/n_possible
end

function is_terminal(history)
    if history[end-1:end] == "cc"
        # round ends when both players check (pass)
        return true 
    elseif history[end-1:end] == "bb" 
        # round ends when both players bet (bets are called)
        return true
    elseif history[end-1:end] == "bc"
        # round ends when a player folds after a bet
        return true
    else 
        return false
    end
end

function terminal_util(history, card_1, card_2)
    n = length(history)
    if n % 2 ==0 # card 1 is player 1
        card_player = card_1
        card_opponent = card_2
    else # player 2 turn, card 1 is player 2
        card_player = card_2
        card_opponent = card_1
    end

    if history[end-1:end] == "bc"
        # fold after bet (no showdown, current player wins 1)
        return 1.0
    elseif history[end-1:end] == "cc"
        # two checks, high card wins
        if card_player > card_opponent
            return 1.0
        else 
            return -1.0
        end
    elseif history[end-1:end] == "bb"
        # double bet 
        if card_player > card_opponent
            return 2.0
        else 
            return -2.0
        end
    end
end

function displayStrategy(i_map)
    # sorted = sort(i_map)
    i_map = sort(i_map)

    
    println("Player 1 Strategies")
    println("Infoset [Pass, Bet]")
    for (key, infoset) in i_map
        # println(key," ", infoset)
        strat = get_average_strategy(infoset)
        if length(key) % 2 !== 0
            
            println(key, " : [", round(strat[1], digits = 3), ", ", round(strat[2], digits = 3),"]")
        end
    end 
    println(" ")
    println("Player 2 Strategies")
    println("Infoset [Pass, Bet]")
    for (key, infoset) in i_map
        strat = get_average_strategy(infoset)
        if length(key) % 2 == 0
            println(key, " : [", round(strat[1], digits = 3), ", ", round(strat[2], digits = 3),"]")
        end
    end
end



function train(i_map, n_iterations)

egv = 0

    for i in 1:n_iterations 
        egv += cfr(i_map) 
        for (key, infoset) in i_map
            # println(key," ", infoset)
            next_strategy(infoset)
        end 
        # println(i)
    end
   
    displayStrategy(i_map)
    # return i_map
end

i_map = Dict()
n_iterations = 100000
train(i_map, n_iterations)
