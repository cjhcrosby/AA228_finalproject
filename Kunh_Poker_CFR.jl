using Random
using StatsBase
using Printf

mutable struct Node
    key::String
    n_actions::Int
    action_dict::Vector{Char}
    possible_actions::Vector{Int}

    regret_sum::Vector{Float64}
    strategy_sum::Vector{Float64}

    strategy::Vector{Float64}
    average_strategy::Vector{Float64}

    function Node(key::String, action_dict::Vector{Char}, n_actions::Int=2)
        self = new()
        self.key = key
        self.n_actions = n_actions
        self.action_dict = action_dict
        self.possible_actions = collect(1:n_actions)

        self.regret_sum = zeros(Float64, self.n_actions)
        self.strategy_sum = zeros(Float64, self.n_actions)

        self.strategy = fill(1.0 / self.n_actions, self.n_actions)
        self.average_strategy = fill(1.0 / self.n_actions, self.n_actions)
        return self
    end
end

function get_strategy(node::Node)
    # regrets are set to zero in cfr+
    node.regret_sum = max.(node.regret_sum, 0.0)
    normalizing_sum = sum(node.regret_sum)
    node.strategy = node.regret_sum
    if normalizing_sum > 0.0
        node.strategy .= node.strategy ./ normalizing_sum
    else
        node.strategy .= fill(1.0 / node.n_actions, node.n_actions)
    end
    return node.strategy
end

function get_action(node::Node, strategy::Vector{Float64})
    return sample(node.possible_actions, Weights(strategy))
end

function get_average_strategy(node::Node)
    strategy = node.strategy_sum

    normalizing_sum = sum(strategy)
    if normalizing_sum > 0.0
        strategy = strategy ./ normalizing_sum
    else
        strategy = fill(1.0 / node.n_actions, node.n_actions)
    end
    return strategy
end

function Base.show(io::IO, node::Node)
    strategies = [ @sprintf("%0.2f", x) for x in get_average_strategy(node) ]
    println(io, "$(lpad(node.key, 6)) $strategies")
end

mutable struct Kunh
    nodeMap::Dict{String, Node}
    expected_game_value::Float64
    n_cards::Int
    nash_equilibrium::Dict{Any, Any}
    current_player::Int
    deck::Vector{Int}
    n_actions::Int
    iters::Int
    AVERAGE_TYPE::String

    function Kunh()
        self = new()
        self.nodeMap = Dict{String, Node}()
        self.expected_game_value = 0.0
        self.n_cards = 3
        self.nash_equilibrium = Dict{Any, Any}()
        self.current_player = 0
        self.deck = [0, 1, 2]
        self.n_actions = 2
        self.iters = 0
        self.AVERAGE_TYPE = "simple"
        return self
    end
end

function train(self::Kunh, n_iterations::Int = 10000)
    expected_game_value = 0.0
    for _ in 1:n_iterations
        self.iters += 1
        # Regrets strategies half way through
        if self.iters == div(n_iterations, 2)
            for (key, v) in self.nodeMap
                v.strategy_sum .= zeros(v.n_actions)
                expected_game_value = 0.0
            end
        end

        for j in 0:1
            self.current_player = j
            shuffle!(self.deck)
            expected_game_value += cfr(self, "")
            if self.AVERAGE_TYPE == "full"
                update_average(self, "", [1.0, 1.0])
            end
        end
    end
    expected_game_value /= n_iterations
    display_results(expected_game_value, self.nodeMap)
end

function cfr(self::Kunh, history::String)
    n = length(history)
    player = n % 2
    player_card = (player == 0) ? self.deck[1] : self.deck[2]

    if is_terminal(history)
        card_opponent = (player == 0) ? self.deck[2] : self.deck[1]
        reward = get_reward(history, player_card, card_opponent)
        return reward
    end

    node = get_node(self, player_card, history)
    strategy = get_strategy(node)
    action_utils = zeros(Float64, self.n_actions)
    if player == self.current_player
        # Counterfactual utility per action.
        for act in 1:self.n_actions
            next_history = history * node.action_dict[act]
            action_utils[act] = -1 * cfr(self, next_history)
        end
        util = sum(action_utils .* strategy)
        regrets = action_utils .- util
        node.regret_sum .= node.regret_sum .+ regrets
        return util
    else
        a = get_action(node, strategy)
        next_history = history * node.action_dict[a]
        util = -1 * cfr(self, next_history)
        if self.AVERAGE_TYPE == "simple"
            node.strategy_sum .= node.strategy_sum .+ strategy
        end
        return util
    end
end

function update_average(self::Kunh, history::String, reach_probs::Vector{Float64})
    n = length(history)
    player = n % 2
    player_card = (player == 0) ? self.deck[1] : self.deck[2]
    if is_terminal(history)
        return
    end
    # If all the probs are zero, zero strategy accumulated, so just return
    if sum(reach_probs) == 0.0
        return
    end
    node = get_node(self, player_card, history)
    strategy = node.strategy
    if player == self.current_player
        new_reach_probs = copy(reach_probs)
        for act in 1:self.n_actions
            new_reach_probs[player + 1] *= strategy[act]
            next_history = history * node.action_dict[act]
            update_average(self, next_history, new_reach_probs)
        end
        node.strategy_sum .= node.strategy_sum .+ reach_probs[player + 1] * strategy
    else
        a = get_action(node, strategy)
        next_history = history * node.action_dict[a]
        reach_probs[player + 1] *= strategy[a]
        update_average(self, next_history, reach_probs)
    end
end

function get_node(self::Kunh, card::Int, history::String)
    key = string(card) * " " * history
    if !haskey(self.nodeMap, key)
        action_dict = ['p', 'b']
        info_set = Node(key, action_dict)
        self.nodeMap[key] = info_set
        return info_set
    end
    return self.nodeMap[key]
end

function get_reward(history::String, player_card::Int, opponent_card::Int)
    terminal_pass = endswith(history, 'p')
    double_bet = endswith(history, "bb")
    if terminal_pass
        if endswith(history, "pp")
            return player_card > opponent_card ? 1 : -1
        else
            return 1
        end
    elseif double_bet
        return player_card > opponent_card ? 2 : -2
    end
    return 0 # In case none of the conditions are met
end

function is_terminal(history::String)
    if length(history) >= 2
        last_two = history[end-1:end]
        return last_two == "pp" || last_two == "bb" || last_two == "bp"
    else
        return false
    end
end

function display_results(ev::Float64, i_map::Dict{String, Node})
    println("player 1 expected value: $(ev)")
    println("player 2 expected value: $(-ev)")
    println()
    println("player 1 strategies:")

    sorted_keys = sort(collect(keys(i_map)))
    for key in sorted_keys
        if iseven(length(key))
            println(i_map[key])
        end
    end
    println()
    println("player 2 strategies:")
    for key in sorted_keys
        if isodd(length(key))
            println(i_map[key])
        end
    end
end

# Main execution
time1 = time()
trainer = Kunh()
train(trainer, 25000)
println(abs(time1 - time()))
