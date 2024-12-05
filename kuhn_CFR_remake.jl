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

function get_strategy(self::Node)
    # Set negative regrets to zero
    self.strategy = max.(self.regret_sum, 0.0)
    normalizing_sum = sum(self.strategy)
    if normalizing_sum > 0.0
        self.strategy /= normalizing_sum
    else
        self.strategy .= 1.0 / self.n_actions
    end
    return self.strategy
end

function get_action(self::Node, strategy::Vector{Float64})
    return sample(self.possible_actions, Weights(strategy))
end

function get_average_strategy(self::Node)
    strategy = self.strategy_sum
    normalizing_sum = sum(strategy)
    if normalizing_sum > 0.0
        strategy /= normalizing_sum
    else
        strategy .= 1.0 / self.n_actions
    end
    return strategy
end

function Base.show(io::IO, self::Node)
    strategies = [ @sprintf("%0.2f", x) for x in get_average_strategy(self) ]
    println(io, "$(lpad(self.key, 6)) $strategies")
end

mutable struct Kunh
    nodeMap::Dict{String, Node}
    current_player::Int
    deck::Vector{Int}
    n_actions::Int
    epsilon::Float64

    function Kunh()
        self = new()
        self.nodeMap = Dict{String, Node}()
        self.current_player = 0
        self.deck = [0, 1, 2]
        self.n_actions = 2
        self.epsilon = 0.14  # Exploration parameter
        return self
    end
end

function sample_strategy(self::Kunh, strategy::Vector{Float64})
    mixed_strategy = self.epsilon * fill(1.0 / self.n_actions, self.n_actions) +
                     (1 - self.epsilon) * strategy
    return mixed_strategy
end

function train(self::Kunh, n_iterations::Int=50000)
    for i in 1:n_iterations
        if i % 1000 == 0
            # println(i)
        end
        # Reset strategy sums halfway through
        if i == div(n_iterations, 2)
            for (key, node) in self.nodeMap
                node.strategy_sum .= zeros(node.n_actions)
            end
        end
        shuffle!(self.deck)
        for j in 0:1
            self.current_player = j
            cfr(self, "", 1.0, 1.0, 1.0)
        end
    end
    display_results(self.nodeMap)
end

function cfr(self::Kunh, history::String, p1_reach::Float64, p2_reach::Float64, sample_reach::Float64)
    n = length(history)
    player = n % 2
    player_card = (player == 0) ? self.deck[1] : self.deck[2]

    if is_terminal(history)
        opponent_card = (player == 0) ? self.deck[2] : self.deck[1]
        reward = get_reward(history, player_card, opponent_card)
        return reward / sample_reach, 1.0
    end

    node = get_node(self, player_card, history)
    strategy = get_strategy(node)

    if player == self.current_player
        # Epsilon-greedy exploration
        probability = sample_strategy(self, strategy)
    else
        probability = strategy
    end

    act = get_action(node, probability)
    next_history = history * node.action_dict[act]

    if player == 0
        util, p_tail = cfr(self, next_history, p1_reach * strategy[act], p2_reach, sample_reach * probability[act])
    else
        util, p_tail = cfr(self, next_history, p1_reach, p2_reach * strategy[act], sample_reach * probability[act])
    end
    util *= -1

    my_reach = (player == 0) ? p1_reach : p2_reach
    opp_reach = (player == 0) ? p2_reach : p1_reach

    if player == self.current_player
        W = util * opp_reach
        for a in 1:self.n_actions
            if a == act
                regret = W * (1.0 - strategy[act]) * p_tail
            else
                regret = -W * p_tail * strategy[act]
            end
            node.regret_sum[a] += regret
        end
    else
        for a in 1:self.n_actions
            node.strategy_sum[a] += (my_reach * strategy[a]) / sample_reach
        end
    end

    return util, p_tail * strategy[act]
end

function get_node(self::Kunh, card::Int, history::String)
    key = string(card) * " " * history
    if !haskey(self.nodeMap, key)
        action_dict = ['p', 'b']
        node = Node(key, action_dict)
        self.nodeMap[key] = node
        return node
    end
    return self.nodeMap[key]
end

function is_terminal(history::String)
    length(history) >= 2 && (endswith(history, "pp") || endswith(history, "bb") || endswith(history, "bp"))
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
    return 0  # Default case
end

function display_results(nodeMap::Dict{String, Node})
    println("Player 1 strategies:")
    sorted_keys = sort(collect(keys(nodeMap)))
    for key in sorted_keys
        if iseven(length(key))
            println(nodeMap[key])
        end
    end
    println()
    println("Player 2 strategies:")
    for key in sorted_keys
        if isodd(length(key))
            println(nodeMap[key])
        end
    end
end

# Exploitability calculation

mutable struct Exploitability
    nodeMap::Dict{String, Node}
    expected_game_value::Float64
    n_cards::Int
    action_dict::Dict{Int, Char}
    average_strategies::Dict{String, Vector{Float64}}
    deck::Vector{Int}
    n_actions::Int
    traverser::Int

    function Exploitability()
        self = new()
        self.nodeMap = Dict{String, Node}()
        self.expected_game_value = 0.0
        self.n_cards = 3
        self.action_dict = Dict(0 => 'p', 1 => 'b')
        # Pre-computed average strategies with CFR
        self.average_strategies = Dict(
            "0 "   => [0.8, 0.2], "0 pb" => [1.0, 0.0],
            "1 "   => [1.0, 0.0], "1 pb" => [0.47, 0.53],
            "2 "   => [0.4, 0.6], "2 pb" => [0.0, 1.0],
            "0 b"  => [1.0, 0.0], "0 p"  => [0.67, 0.33],
            "1 b"  => [0.67, 0.33], "1 p"  => [1.0, 0.0],
            "2 b"  => [0.0, 1.0], "2 p"  => [0.0, 1.0]
        )
        self.deck = [0, 1, 2]
        self.n_actions = 2
        self.traverser = 1
        return self
    end
end

function walk_tree(self::Exploitability, history::String, pr_2::Vector{Float64})
    n = length(history)
    player = n % 2
    if is_terminal(history)
        return get_reward_exploitability(self, history, pr_2)
    end
    strategies = [ self.average_strategies[string(card) * " " * history] for card in 0:(self.n_cards - 1) ]
    utils = zeros(Float64, self.n_cards)
    action_utils = zeros(Float64, self.n_actions, self.n_cards)
    for action in 1:self.n_actions
        next_history = history * self.action_dict[action - 1]
        strategies_taken = [ strategies[card + 1][action] for card in 0:(self.n_cards - 1) ]
        if self.traverser == player
            action_utils[action, :] .= walk_tree(self, next_history, pr_2)
        else
            action_utils[action, :] .= walk_tree(self, next_history, pr_2 .* strategies_taken)
            utils .+= action_utils[action, :]
        end
    end
    if player == self.traverser
        # Pass up the best response
        for i in 1:self.n_cards
            utils[i] = maximum(action_utils[:, i])
        end
    end
    return utils
end

function get_reward_exploitability(self::Exploitability, history::String, opp_reach::Vector{Float64})
    terminal_pass = endswith(history, 'p')
    double_bet = endswith(history, "bb")
    if terminal_pass
        if endswith(history, "pp")
            return show_down(self, opp_reach, 1)
        else
            fold_player = who_folded(history)
            return fold_reward(self, opp_reach, fold_player)
        end
    elseif double_bet
        return show_down(self, opp_reach, 2)
    end
    return zeros(Float64, self.n_cards)  # Default case
end

function who_folded(history::String)
    if length(history) == 3
        return 0  # Player 0 folded
    elseif length(history) == 2
        return 1  # Player 1 folded
    end
    return -1  # Should not reach here
end

function show_down(self::Exploitability, opp_reach::Vector{Float64}, payoff::Int)
    total_prob = sum(opp_reach)
    p_lose = total_prob
    p_win = 0.0
    reward = zeros(Float64, self.n_cards)
    for card in 0:(self.n_cards - 1)
        p_lose -= opp_reach[card + 1]
        reward[card + 1] = (p_win - p_lose) * payoff
        p_win += opp_reach[card + 1]
    end
    return reward
end

function fold_reward(self::Exploitability, opp_reach::Vector{Float64}, fold_player::Int)
    payoff = (fold_player == self.traverser) ? -1 : 1
    loss_prob = sum(opp_reach)
    rewards = zeros(Float64, self.n_cards)
    for card in 0:(self.n_cards - 1)
        loss_prob -= opp_reach[card + 1]
        rewards[card + 1] = payoff * loss_prob
        loss_prob += opp_reach[card + 1]
    end
    return rewards
end

# Main execution
time1 = time()
trainer = Kunh()
train(trainer, 100000)
println("Training time: ", abs(time() - time1), " seconds")

# Calculate exploitability
exploit_calc = Exploitability()
startingReaches = ones(Float64, exploit_calc.n_cards)
exploit_calc.traverser = 0
p1_average_ev = sum(walk_tree(exploit_calc, "", startingReaches))
println("Player 1 exploitability: ", p1_average_ev)

startingReaches = ones(Float64, exploit_calc.n_cards)
exploit_calc.traverser = 1
p2_average_ev = sum(walk_tree(exploit_calc, "", startingReaches))
println("Player 2 exploitability: ", p2_average_ev)
println("Average exploitability: ", (p1_average_ev + p2_average_ev) / 2)
