#= 
AA 228 Final Project 
Colton Crosby, William Ho, Kevin Murillo
=#

using Random
using Distributions
using DataStructures
using Plots
using StatsBase
using PrettyTables


Actions = ["p","b"] # pass, bet
const NUM_ACTIONS = length(Actions)
const NUM_PLAYERS = 2

# deck functions
mutable struct KuhnDeck
    cards::Vector{String}
    index::Int
end

function create_kuhn_deck()::KuhnDeck
    ranks = ["K", "Q", "J"]
    suits = ["♠"]
    cards = [rank * suit for rank in ranks for suit in suits]
    shuffle!(cards)
    return KuhnDeck(cards, 1)
end

function draw!(deck::KuhnDeck, num_cards::Int = 1)::Vector{String}
    if deck.index + num_cards - 1 > length(deck.cards)
        error("Not enough cards left in the deck")
    end
    drawn_cards = deck.cards[deck.index : deck.index + num_cards - 1]
    deck.index += num_cards
    return drawn_cards
end

function shuffleDeck!(deck::KuhnDeck)
    shuffle!(deck.cards)
    deck.index = 1  # Reset index to the start
end


# # # # # # # # # # #
mutable struct History
    player_cards::Vector{String} # Cards dealt to players
    actions::Vector{String}      # Sequence of actions taken (e.g., "p", "b")
    current_player
end

function initialize_history(deck::KuhnDeck)::History
    cards = draw!(deck, 2) # Deal one card to each player
    return History(cards, String[], 1)
end

function is_terminal(history::History)::Bool
    # A terminal history occurs when:
    # - Both players pass
    # - One player bets and the other folds (i.e., passes after a bet)
    # - A showdown occurs (both players bet)
    actions = history.actions
    # @info history
    length(actions) >= 2 && (
        (actions[end] == "p" && actions[end-1] == "p") || # both pass (showdown)
        (actions[end] == "p" && actions[end-1] == "b") || # second player folds (pass after bet)
        (actions[end] == "b" && actions[end-1] == "b") || # both bet
        (length(actions) >= 4) # game runs over (shouldn't happen)
    )
end


function utility(history::History)::Vector{Float64}
    # Calculate the utility for players in terminal history
    if !is_terminal(history)
        error("Utility can only be calculated for terminal histories.")
    end
    card_vals = Dict("K♠" => 3, "Q♠" => 2, "J♠" => 1)
    hact = history.actions
    terminal_pass = hact[end] == "p"
    double_bet = join(hact[end-1:end]) == "bb"
    p1_card = history.player_cards[1]
    p2_card = history.player_cards[2]
    p1_val = card_vals[p1_card]
    p2_val = card_vals[p2_card]
    # terminal_pass = hact[end] == "p"
    # println("P1: ", p1_card, ", P2: ", p2_card, ", actions: ",hact)
    if terminal_pass 
        if join(hact[end-1:end]) == "pp"
            if p1_val > p2_val
                util = [1.0, -1.0]
            else
                util = [-1.0, 1.0]
            end
        else
            util = [1.0, -1.0]
        end
    elseif double_bet
        if p1_val > p2_val
            util = [2.0, -2.0]
        else
            util = [-2.0, 2.0]
        end
    else
        error("unexpected terminal history  in utility function")
    end
    # println("outcome: ", util)
    return util
end


# # # # # # # 

mutable struct InfoSet
    card::String                # The card the player holds
    actions::Vector{String}     # Actions taken so far
    regret_sum::Vector{Float64} # Accumulated regrets for each action
    strategy_sum::Vector{Float64} # Sum of strategies over iterations
    strategy::Vector{Float64}
end

function initialize_infoset(card::String, num_actions::Int)::InfoSet
    regret_sum = zeros(Float64, num_actions)
    strategy_sum = zeros(Float64, num_actions)
    return InfoSet(card, String[], regret_sum, strategy_sum, [0.5, 0.5])
end


function get_strategy(infoset::InfoSet)::Vector{Float64}
    # Compute the strategy using regrets
    infoset.regret_sum = max.(infoset.regret_sum, 0.0)
    normalizing_sum = sum(infoset.regret_sum)
    infoset.strategy = infoset.regret_sum
    if normalizing_sum > 0
        infoset.strategy = infoset.strategy / normalizing_sum
    else
        infoset.strategy = fill(1.0 / NUM_ACTIONS, NUM_ACTIONS) # Uniform strategy
    end
    # @info infoset
    return infoset.strategy
end

function get_average_strategy(infoset::InfoSet)::Vector{Float64}
    strategy = infoset.strategy_sum
    normalizing_sum = sum(infoset.strategy_sum)
    if normalizing_sum > 0
        return infoset.strategy_sum / normalizing_sum
    else
        return fill(1.0 / NUM_ACTIONS, NUM_ACTIONS)
    end
end


# # # # # # 



function play_kuhns_poker(infosets::Dict{String, InfoSet}, history::History, reach_prob::Vector{Float64})
    # If the game reaches a terminal state, return utilities
    if is_terminal(history)
        return
    end

    if sum(reach_prob) == 0
        return
    end

    n = length(join(history.actions)) # length(join(history.player_cards)) +
    player = n % 2 + 1

    # Get the infoset key and strategy for the current player
    card = history.player_cards[player] # index from 0 to 1
    infoset_key = card * join(history.actions, "")  # Card + action history as key
    infoset = infosets[infoset_key]
    
    strategy = get_strategy(infoset) # infoset strategy

    if player == history.current_player # if player is agent
        new_reach_prob = copy(reach_prob)
        # simulate taking next actions
        for a_idx in 1:NUM_ACTIONS
            action = Actions[a_idx]
            push!(history.actions, action) # push action to history

            new_reach_prob[player] *= strategy[a_idx] # update reach Probability
            play_kuhns_poker(infosets, history, new_reach_prob)
            pop!(history.actions) # pop action out of history for next sim
        end
        infoset.strategy_sum += reach_prob[player] * strategy
    else # if it is player 2 (opponent)
        a = rand(Categorical(strategy)) # sample an action from the strategy
        # a = rand([1,2]) # sample an action from the strategy
        
        action = Actions[a]
        push!(history.actions, action) # push action to history
        reach_prob[player] *= strategy[a]
        play_kuhns_poker(infosets, history, reach_prob)
        pop!(history.actions) # pop action out of history for next sim
    end
end

function cfr_update(infosets::Dict{String, InfoSet}, history::History)
    n = length(join(history.actions)) #  length(join(history.player_cards)) + 
    player = n % 2 + 1
    card = history.player_cards[player]

    if is_terminal(history)
        # println("showdown strategy: ",)
        term_util = utility(history)
        return term_util[player]
    end
    
    actions = history.actions
    card = history.player_cards[player] # index from 0 to 1
    infoset_key = card * join(history.actions, "")  # Card + action history as key
    infoset = infosets[infoset_key]
    strategy = get_strategy(infoset)
    action_utils = zeros(NUM_ACTIONS)
    # println("showdown strategy: ", strategy)
    if player == history.current_player 
        # calculate counterfactual utility over actions
        for a_idx in 1:NUM_ACTIONS
            action = Actions[a_idx]
            
            push!(history.actions, action)
            # println(history.actions)
            action_utils[a_idx] = -1.0 * cfr_update(infosets, history)
            pop!(history.actions) # pop action out of history for next sim
            # println(history.actions, " popped")  
            # println(action_utils)  
        end
        util = sum(action_utils .* strategy)
        # println(history.player_cards[1],history.actions)
        # println("strat = ", strategy)
        regrets = action_utils .- util 
        # println(regrets," = ", action_utils, " - ", util)
        infoset.regret_sum += regrets 
        # println("P1 util: ", util)
        return util
    else
        a = rand(Categorical(strategy)) # sample an action from the strategy
        # a = rand([1,2])
        action = Actions[a]
        push!(history.actions, action) # push action to history
        util = -1.0 * cfr_update(infosets, history)
        # println("P2 util: ", util)
        pop!(history.actions) 
        infoset.strategy_sum += strategy 
        return util
    end
end


function initialize_infosets(deck::KuhnDeck)::Dict{String, InfoSet}
    infosets = Dict{String, InfoSet}()
    
    # Initialize infosets for each player card and action history
    for card in deck.cards
        # Generate infosets for each possible action history
        for action_history in ["", "p", "b", "pb"]
            key = card * action_history  # Unique key for each infoset (card + action history)
            infosets[key] = initialize_infoset(card, NUM_ACTIONS)
        end
    end
    
    return infosets
end


function print_strategy_table(infosets::Dict{String, InfoSet})
    table = [["Infoset", "Pass", "Bet"]]
    for (key, infoset) in infosets
        strategy = get_average_strategy(infoset)
        # strategy = infoset.strategy_sum
        push!(table, [key, string(round(strategy[1], digits = 3)), string(round(strategy[2], digits = 3))])
    end
    pretty_table(table)
end


function train_kuhn_poker_cfr(iterations::Int)
    EV = 0
    deck = create_kuhn_deck()
    infosets = initialize_infosets(deck)
    for i in 1:iterations
        deck = create_kuhn_deck()  # Reshuffle deck at the start of each game
        history = initialize_history(deck)  # Initialize a fresh history for this game
        if i == iterations ÷ 2
            for (k,v) in infosets 
                v.strategy_sum = zeros(NUM_ACTIONS)
                EV = 0
            end
        end
        for j in 1:NUM_PLAYERS 
            history.current_player = j 
            shuffleDeck!(deck)
            EV += cfr_update(infosets, history)
            println("EV: ",EV)
        end
        # Start recursive CFR from Player 1 with equal probabilities for both players
        play_kuhns_poker(infosets, history, [1.0, 1.0])
        # print_strategy_table(infosets)
        shuffleDeck!(deck)
    end
    EV /= iterations
    println("final EV: ", EV)

    # Output the final strategies
    println("\nFinal Strategies:")
    print_strategy_table(infosets)
end


# function train_kuhn_poker_cfr(iterations::Int)
#     deck = create_kuhn_deck()
#     infosets = initialize_infosets(deck)

#     for i in 1:iterations
#         deck = create_kuhn_deck()  # Reshuffle deck at the start of each game
#         history = initialize_history(deck)  # Initialize a fresh history for this game

#         # Start recursive CFR from Player 1 with equal probabilities for both players
#         play_kuhns_poker(infosets, history, [1.0, 1.0])
#     end

#     # Output the final strategies
#     println("\nFinal Strategies:")
#     print_strategy_table(infosets)
# end


# # # # # # # #
iterations = 10000


train_kuhn_poker_cfr(iterations)
