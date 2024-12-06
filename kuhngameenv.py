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

# Dynamic agent strategy based on card and history
def agent_strategy(card, history):
    if card == "K":
        if history == "":  # First move
            return random.choices(["Pass", "Bet"], weights=[.38, .62])[0]
        elif history == "PassBet":  # Opponent bet
            return random.choices(["Pass", "Bet"], weights=[0, 1.00])[0]
    elif card == "Q":
        if history == "":  # First move
            return random.choices(["Pass", "Bet"], weights=[1.00, 0])[0]
        elif history == "PassBet":  # Opponent bet
            return random.choices(["Pass", "Bet"], weights=[.45, .55])[0]
    elif card == "J":
        if history == "":  # First move
            return random.choices(["Pass", "Bet"], weights=[.80, .20])[0]
        elif history == "PassBet":  # Opponent bet
            return random.choices(["Pass", "Bet"], weights=[1.00, 0])[0]

# Dynamic opponent strategy based on card and history
def opponent_strategy(card, history):
    if card == "K":
        if history == "Pass":  # P1 passed
            return random.choices(["Pass", "Bet"], weights=[0, 100])[0]
        elif history == "Bet":  # P1 bet
            return random.choices(["Pass", "Bet"], weights=[0, 100])[0]
    elif card == "Q":
        if history == "Pass":  # P1 passed
            return random.choices(["Pass", "Bet"], weights=[100, 0])[0]
        elif history == "Bet":  # P1 bet
            return random.choices(["Pass", "Bet"], weights=[67, 33])[0]
    elif card == "J":
        if history == "Pass":  # P1 passed
            return random.choices(["Pass", "Bet"], weights=[67, 33])[0]
        elif history == "Bet":  # P1 bet
            return random.choices(["Pass", "Bet"], weights=[100, 0])[0]

# Deal cards for Kuhn Poker
def deal_cards():
    cards = ["J", "Q", "K"]
    random.shuffle(cards)
    return cards[0], cards[1]

# Evaluate the winner of the game
def evaluate_winner(state):
    card_rank = {"J": 1, "Q": 2, "K": 3}
    # If both players pass, determine winner by card rank
    if state.history.endswith("PassPass"):
        return "opponent" if card_rank[state.opponent_card] > card_rank[state.player_card] else "player"
    # If both players bet, determine winner by card rank
    if "BetBet" in state.history:
        return "opponent" if card_rank[state.opponent_card] > card_rank[state.player_card] else "player"
    # If P1 folds after P2's bet
    if state.history.endswith("PassBetPass"):
        return "opponent"
    # If P2 folds after P1's bet
    if state.history.endswith("BetPass"):
        return "player"
    return None  # Should never reach here in valid game states

def play_game(agent_model, player_chips, opponent_chips):
    # Deal cards
    player_card, opponent_card = deal_cards()
    state = GameState(player_card, opponent_card, history="")  # Clear history for new round

    # Ante
    player_chips -= 1
    opponent_chips -= 1
    state.pot += 2

    actions = []  # Track actions for this round

    # Game loop
    while True:
        if state.current_player == "P1":
            # Player 1's action
            player_action = agent_strategy(state.player_card, state.history)
            actions.append(("P1", player_action))
            state.history += player_action
            if player_action == "Bet":
                if "PassBetBet" in state.history:  # Player 1 calls Player 2's bet
                    player_chips -= 1  # Deduct 1 chip for the call
                    state.pot += 1  # Add Player 1's chip to the pot
                    break  # Go to showdown
                player_chips -= 1  # Deduct 1 chip from Player 1
                state.pot += 1  # Add 1 chip to the pot
                state.current_player = "P2"  # Move to Player 2
            elif player_action == "Pass":
                if "PassBetPass" in state.history:
                    break  # Player 1 folds after Player 2's bet; Player 2 wins
                if "PassPass" in state.history:
                    break  # Both players pass; showdown
                state.current_player = "P2"  # Check; move to Player 2
        else:
            # Player 2's action
            opponent_action = opponent_strategy(state.opponent_card, state.history)
            actions.append(("P2", opponent_action))
            state.history += opponent_action
            agent_model.update(state.opponent_card, state.history, opponent_action)
            if opponent_action == "Bet":
                opponent_chips -= 1  # Deduct 1 chip from Player 2
                state.pot += 1  # Add 1 chip to the pot
                if "Bet" in state.history and state.history.endswith("BetBet"):
                    break  # Both players have bet; showdown
                if "PassBetPass" in state.history:
                    break  # Player 1 folds after Player 2's bet; Player 2 wins
                state.current_player = "P1"  # Move back to Player 1
            elif opponent_action == "Pass":
                if "PassPass" in state.history:
                    break  # Both players pass; showdown
                if "BetPass" in state.history:
                    break  # Player 2 folds; Player 1 wins
                state.current_player = "P1"  # Check; move back to Player 1

    # Determine winner and update chips
    winner = evaluate_winner(state)
    if winner == "player":
        player_chips += state.pot
    else:
        opponent_chips += state.pot

    return winner, player_card, opponent_card, actions, player_chips, opponent_chips



# Train the agent by playing multiple games
def train_agent(num_games, starting_chips):
    agent_model = OpponentModel()
    player_chips = starting_chips
    opponent_chips = starting_chips
    results = {"player": 0, "opponent": 0}
    round_logs = []  # Store logs for each round

    for _ in range(num_games):
        winner, player_card, opponent_card, actions, player_chips, opponent_chips = play_game(agent_model, player_chips, opponent_chips)

        results[winner] += 1
        round_logs.append({
            "player_card": player_card,
            "opponent_card": opponent_card,
            "winner": winner,
            "actions": actions,
            "player_chips": player_chips,
            "opponent_chips": opponent_chips,
        })

    return agent_model, results, player_chips, opponent_chips, round_logs

# Main simulation
def main():
    num_games = 100000
    starting_chips = 200000  # Starting chips for each player
    agent_model, results, player_chips, opponent_chips, round_logs = train_agent(num_games, starting_chips)

    print(f"Player's net wins after {num_games} games: {results['player'] - results['opponent']}")
    print(f"Player's net chips {(player_chips - starting_chips)}")   
    print(f"Player's chip win rate: {(player_chips - starting_chips) / num_games:.2f}")
    print(f"Player's chips after {num_games} games: {player_chips}")
    print(f"Opponent's chips after {num_games} games: {opponent_chips}")
'''
    # Display each round's details
    print("\nRound Logs:")
    for i, log in enumerate(round_logs, 1):
        print(f"Round {i}:")
        print(f"  Player Card: {log['player_card']}")
        print(f"  Opponent Card: {log['opponent_card']}")
        print(f"  Winner: {log['winner']}")
        print(f"  Actions: {log['actions']}")
        print(f"  Player Chips: {log['player_chips']}")
        print(f"  Opponent Chips: {log['opponent_chips']}\n")
        '''

if __name__ == "__main__":
    main()
