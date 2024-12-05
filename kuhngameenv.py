import random
from collections import defaultdict

# Define the game state structure
class GameState:
    def __init__(self, player_card, opponent_card, history="", pot=0, current_player="P1"):
        self.player_card = player_card
        self.opponent_card = opponent_card
        self.history = history
        self.pot = pot  # Pot starts with 0 and grows with antes and bets
        self.current_player = current_player

# Define the opponent move tracker
class OpponentModel:
    def __init__(self):
        # Card -> History -> Action -> Count
        self.action_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def update(self, card, history, action):
        self.action_counts[card][history][action] += 1

# Define probabilities for actions based on the card
def action_probabilities(card):
    if card == "K":
        return {"Pass": 0.1, "Bet": 0.9}  # Aggressive with K
    elif card == "Q":
        return {"Pass": 0.5, "Bet": 0.5}  # Neutral with Q
    elif card == "J":
        return {"Pass": 0.9, "Bet": 0.1}  # Defensive with J

# Agent strategy (probabilistic based on the card)
def agent_strategy(card, history):
    probs = action_probabilities(card)
    return random.choices(["Pass", "Bet"], weights=[probs["Pass"], probs["Bet"]])[0]

# Opponent strategy (probabilistic based on the card)
def opponent_strategy(card, history):
    probs = action_probabilities(card)
    return random.choices(["Pass", "Bet"], weights=[probs["Pass"], probs["Bet"]])[0]

# Deal cards for Kuhn Poker
def deal_cards():
    cards = ["J", "Q", "K"]
    random.shuffle(cards)
    return cards[0], cards[1]

# Evaluate the winner of the game
def evaluate_winner(state):
    card_rank = {"J": 1, "Q": 2, "K": 3}
    if "Pass" in state.history and state.history.endswith("Pass"):
        return "opponent" if state.current_player == "P1" else "player"  # Last actor loses
    return "player" if card_rank[state.player_card] > card_rank[state.opponent_card] else "opponent"

# Simulate a single game
def play_game(agent_model, player_chips, opponent_chips):
    # Deal cards
    player_card, opponent_card = deal_cards()
    state = GameState(player_card, opponent_card)

    # Ante
    player_chips -= 1
    opponent_chips -= 1
    state.pot += 2

    # Game loop
    while True:
        if state.current_player == "P1":
            # Player 1's action
            player_action = agent_strategy(state.player_card, state.history)
            state.history += player_action
            if player_action == "Bet":
                player_chips -= 1
                state.pot += 1
            if "Pass" in state.history or state.history.endswith("BetBet"):
                break
            state.current_player = "P2"
        else:
            # Player 2's action
            opponent_action = opponent_strategy(state.opponent_card, state.history)
            state.history += opponent_action
            agent_model.update(state.opponent_card, state.history, opponent_action)
            if opponent_action == "Bet":
                opponent_chips -= 1
                state.pot += 1
            if "Pass" in state.history or state.history.endswith("BetBet"):
                break
            state.current_player = "P1"

    # Determine winner and update chips
    winner = evaluate_winner(state)
    if winner == "player":
        player_chips += state.pot
    else:
        opponent_chips += state.pot

    return winner, player_chips, opponent_chips

# Train the agent by playing multiple games
def train_agent(num_games, starting_chips):
    agent_model = OpponentModel()
    player_chips = starting_chips
    opponent_chips = starting_chips
    results = {"player": 0, "opponent": 0}

    for _ in range(num_games):
        winner, player_chips, opponent_chips = play_game(agent_model, player_chips, opponent_chips)
        results[winner] += 1

    return agent_model, results, player_chips, opponent_chips

# Display the opponent's move counts
def display_opponent_moves(model):
    print("\nOpponent's Move Counts (P2):")
    for card, histories in model.action_counts.items():
        print(f"Opponent Card: {card}")
        for history, actions in histories.items():
            print(f"  History '{history}': {dict(actions)}")

# Main simulation
def main():
    num_games = 1
    starting_chips = 100  # Starting chips for each player
    agent_model, results, player_chips, opponent_chips = train_agent(num_games, starting_chips)

    print(f"Player's net wins after {num_games} games: {results['player'] - results['opponent']}")
    print(f"Player's chips after {num_games} games: {player_chips}")
    print(f"Opponent's chips after {num_games} games: {opponent_chips}")
    display_opponent_moves(agent_model)

if __name__ == "__main__":
    main()
