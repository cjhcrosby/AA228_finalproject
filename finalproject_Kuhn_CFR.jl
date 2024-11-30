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
    current_player::Int          # Index of the player whose turn it is
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

    actions = history.actions
    if actions[end] == "p" && actions[end-1] == "b"
        # Player 2 folds after Player 1 bets
        return [1.0, -1.0]
    elseif actions[end] == "p" && actions[end-1] == "p"
        # Both players pass (showdown)
        card_vals = Dict("K♠" => 3, "Q♠" => 2, "J♠" => 1)
        winner = argmax([card_vals[history.player_cards[1]], card_vals[history.player_cards[2]]])
        payoff = 1.0
        return winner == 1 ? [payoff, -payoff] : [-payoff, payoff]
    elseif actions[end] == "b" && actions[end-1] == "b"
        # Both players bet (showdown)
        card_vals = Dict("K♠" => 3, "Q♠" => 2, "J♠" => 1)
        winner = argmax([card_vals[history.player_cards[1]], card_vals[history.player_cards[2]]])
        payoff = 2.0
        return winner == 1 ? [payoff, -payoff] : [-payoff, payoff]
    else
        error("Unexpected terminal history.")
    end
end


# # # # # # # 

mutable struct InfoSet
    card::String                # The card the player holds
    actions::Vector{String}     # Actions taken so far
    regret_sum::Vector{Float64} # Accumulated regrets for each action
    strategy_sum::Vector{Float64} # Sum of strategies over iterations
end

function initialize_infoset(card::String, num_actions::Int)::InfoSet
    regret_sum = zeros(Float64, num_actions)
    strategy_sum = zeros(Float64, num_actions)
    return InfoSet(card, String[], regret_sum, strategy_sum)
end



function get_strategy(infoset::InfoSet)::Vector{Float64}
    # Compute the strategy using regrets
    positive_regrets = max.(infoset.regret_sum, 0.0)
    normalizing_sum = sum(positive_regrets)
    if normalizing_sum > 0
        return positive_regrets / normalizing_sum
    else
        return fill(1.0 / NUM_ACTIONS, NUM_ACTIONS) # Uniform strategy
    end
end

function get_average_strategy(infoset::InfoSet)::Vector{Float64}
    normalizing_sum = sum(infoset.strategy_sum)
    if normalizing_sum > 0
        return infoset.strategy_sum / normalizing_sum
    else
        return fill(1.0 / NUM_ACTIONS, NUM_ACTIONS)
    end
end


function update_strategy!(infoset::InfoSet, strategy::Vector{Float64})
    infoset.strategy_sum .+= strategy
end

# # # # # # 

# function play_kuhns_poker(deck::KuhnDeck, infosets::Dict{String, InfoSet}, history::History)
#     # Your code logic here for playing the game using the given history
#     # Iterate through the game for the number of iterations or until terminal state is reached
#     for i in 1:5
#         # Player 1's decision
#         player1_card = history.player_cards[1]
#         player1_infoset = infosets[player1_card]
#         player1_strategy = get_strategy(player1_infoset)

#         # Player 1 action
#         player1_action = sample(Actions, Weights(player1_strategy))
#         push!(history.actions, player1_action)

#         if is_terminal(history)
#             break
#         end

#         # Player 2's decision
#         player2_card = history.player_cards[2]
#         player2_infoset = infosets[player2_card]
#         player2_strategy = get_strategy(player2_infoset)

#         # Player 2 action
#         player2_action = sample(Actions, Weights(player2_strategy))
#         push!(history.actions, player2_action)

#         if is_terminal(history)
#             break
#         end
#     end

#     # Calculate the utility at the end of the game
#     return utility(history)
# end

# function play_kuhns_poker(deck::KuhnDeck, infosets::Dict{String, InfoSet}, history::History, player::Int, prob::Vector{Float64})
#     # If the game reaches a terminal state, return utilities
#     # @info history
#     if is_terminal(history)
#         return utility(history)
#     end

#     # Get the infoset key and strategy for the current player
#     card = history.player_cards[player]
#     infoset_key = card * join(history.actions, "")  # Card + action history as key
#     infoset = infosets[infoset_key]
#     strategy = get_strategy(infoset)

#     # Initialize utility and regret values for the current infoset
#     util = zeros(Float64, NUM_ACTIONS)
#     node_util = 0.0

#     # Iterate over possible actions
#     for a_idx in 1:NUM_ACTIONS
#         # Simulate taking action a_idx
#         action = Actions[a_idx]
#         push!(history.actions, action)

#         # Calculate probabilities for the other player
#         new_prob = copy(prob)
#         new_prob[player] *= strategy[a_idx]

#         # Recursively calculate utility
#         util[a_idx] = play_kuhns_poker(deck, infosets, history, 3 - player, new_prob)[player]
#         node_util += strategy[a_idx] * util[a_idx]

#         # Undo action
#         pop!(history.actions)
#     end

#     # Update regrets
#     for a_idx in 1:NUM_ACTIONS
#         regret = util[a_idx] - node_util
#         infoset.regret_sum[a_idx] += prob[3 - player] * regret
#     end

#     # Update strategy sum for average strategy
#     update_strategy!(infoset, strategy)

#     return node_util
# end

function play_kuhns_poker(deck::KuhnDeck, infosets::Dict{String, InfoSet}, history::History, player::Int, prob::Vector{Float64})
    # If the game reaches a terminal state, return utilities
    if is_terminal(history)
        return utility(history)
    end

    # Get the infoset key and strategy for the current player
    card = history.player_cards[player]
    infoset_key = card * join(history.actions, "")  # Card + action history as key
    infoset = infosets[infoset_key]
    strategy = get_strategy(infoset)

    # Initialize utility and regret values for the current infoset
    util = zeros(Float64, NUM_ACTIONS)
    node_util = 0.0

    # Iterate over possible actions
    for a_idx in 1:NUM_ACTIONS
        # Simulate taking action a_idx
        action = Actions[a_idx]
        push!(history.actions, action)

        # Calculate probabilities for the other player
        new_prob = copy(prob)
        new_prob[player] *= strategy[a_idx]

        # Recursively calculate utilities for both players
        sub_util = play_kuhns_poker(deck, infosets, history, 3 - player, new_prob)
        util[a_idx] = sub_util[player]
        node_util += strategy[a_idx] * sub_util[player]

        # Undo action
        pop!(history.actions)
    end

    # Update regrets
    for a_idx in 1:NUM_ACTIONS
        regret = util[a_idx] - node_util
        infoset.regret_sum[a_idx] += prob[3 - player] * regret
    end

    # Update strategy sum for average strategy
    update_strategy!(infoset, strategy)

    # Return a vector of utilities for both players
    utilities = zeros(Float64, 2)
    utilities[player] = node_util
    utilities[3 - player] = -node_util  # Zero-sum game: u1 + u2 = 0
    return utilities
end



# Function to perform CFR updates
function cfr_update(infosets::Dict{String, InfoSet}, history::History, utilities::Vector{Float64}, cfr_weight::Float64)
    actions = history.actions
    player1_card = history.player_cards[1]
    player2_card = history.player_cards[2]
    
    # Update regrets for Player 1
    player1_infoset = infosets[player1_card]
    player1_strategy = get_strategy(player1_infoset)
    for action in Actions
        regret = utilities[1] - utilities[2]  # Regret for Player 1
        idx = findfirst(x -> x == action, Actions)
        player1_infoset.regret_sum[idx] += cfr_weight * regret
    end
    
    # Update regrets for Player 2
    player2_infoset = infosets[player2_card]
    player2_strategy = get_strategy(player2_infoset)
    for action in Actions
        regret = utilities[2] - utilities[1]  # Regret for Player 2
        idx = findfirst(x -> x == action, Actions)
        player2_infoset.regret_sum[idx] += cfr_weight * regret
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


# function print_strategy_table(infosets::Dict{String, InfoSet})
#     # Initialize the table with column headers
#     table = [["Infoset", "Bet", "Pass"]]
    
#     # Iterate over each infoset
#     for (key, infoset) in infosets
#         # Extract the strategy for each infoset
#         strategy = get_strategy(infoset)
        
#         # Add a row to the table: infoset, bet strategy, pass strategy
#         push!(table, [key, string(strategy[1]), string(strategy[2])])
#     end

#     # Print the table
#     pretty_table(table)
# end

function print_strategy_table(infosets::Dict{String, InfoSet})
    table = [["Infoset", "Bet", "Pass"]]
    for (key, infoset) in infosets
        strategy = get_average_strategy(infoset)
        push!(table, [key, string(round(strategy[2], digits = 3)), string(round(strategy[1], digits = 3))])
    end
    pretty_table(table)
end


# function train_kuhn_poker_cfr(iterations::Int)
#     deck = create_kuhn_deck()
#     infosets = initialize_infosets(deck)
    
#     for i in 1:iterations
#         deck = create_kuhn_deck()  # Reshuffle deck at the start of each game
#         history = initialize_history(deck)  # Initialize a fresh history for this game
#         utilities = play_kuhns_poker(deck, infosets, history)  # Play a game and get utilities
        
#         cfr_weight = 1.0 / iterations  # Weight for CFR updates
#         cfr_update(infosets, history, utilities, cfr_weight)  # Perform CFR update after each game
        
#         # After each iteration, output the current strategies in a table format
#         println("\nIteration $i Strategies:")
#         print_strategy_table(infosets)
#     end
# end

function train_kuhn_poker_cfr(iterations::Int)
    deck = create_kuhn_deck()
    infosets = initialize_infosets(deck)

    for i in 1:iterations
        deck = create_kuhn_deck()  # Reshuffle deck at the start of each game
        history = initialize_history(deck)  # Initialize a fresh history for this game

        # Start recursive CFR from Player 1 with equal probabilities for both players
        play_kuhns_poker(deck, infosets, history, 1, [1.0, 1.0])
    end

    # Output the final strategies
    println("\nFinal Strategies:")
    print_strategy_table(infosets)
end


# # # # # # # #
iterations = 100000


train_kuhn_poker_cfr(iterations)
# deck = create_kuhn_deck()
#     infosets = initialize_infosets(deck)
#     print_strategy_table(infosets)
# deck = create_kuhn_deck()  # Reshuffle deck at the start of each game
# history = initialize_history(deck)  # Initialize a fresh history for this game
# infosets = initialize_infosets(deck)
# utilities = play_kuhns_poker(deck, infosets, history)  # Play a game and get utilities