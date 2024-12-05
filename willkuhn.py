import random

# Constants
CARDS = ["J", "Q", "K"]
PLAYER_ACTIONS = ["check", "bet", "call", "fold"]

# Specify user position: choose Player 1 or Player 2
USER_POSITION = 1  # Set to 1 to play as Player 1, or 2 to play as Player 2

# Function to deal cards
def deal_cards():
    deck = CARDS[:]
    random.shuffle(deck)
    return deck[:2]  # Return two cards (one for each player)

# Play a single hand of Kuhn Poker
def play_kuhn_poker():
    # Initialize players and pot
    player1 = {"name": "Player 1", "card": None, "chips": 10}
    player2 = {"name": "Player 2", "card": None, "chips": 10}
    pot = 0
    
    # Deal cards
    player1["card"], player2["card"] = deal_cards()
    if USER_POSITION == 1:
        print(f"Your card: {player1['card']}.")
        print("Player 2's card is hidden.")
    else:
        print(f"Your card: {player2['card']}.")
        print("Player 1's card is hidden.")
    
    # Ante (each player puts 1 chip in the pot)
    player1["chips"] -= 1
    player2["chips"] -= 1
    pot += 2

    # Betting round
    print("\n--- Betting Round ---")
    if USER_POSITION == 1:
        action1 = get_user_action(player1, "check or bet")
        action2 = get_algorithm_action(player2, action1)
    else:
        action1 = get_algorithm_action(player1, "check or bet")
        action2 = get_user_action(player2, "call or fold" if action1 == "bet" else "check or bet")
    
    # Resolve actions
    if action1 == "bet" and action2 == "call":
        player2["chips"] -= 1
        pot += 1
        print(f"{player2['name']} calls. Pot is now {pot}.")
    elif action1 == "bet" and action2 == "fold":
        print(f"{player2['name']} folds. {player1['name']} wins the pot of {pot} chips!")
        return
    elif action1 == "check" and action2 == "bet":
        response = get_user_action(player1, "call or fold") if USER_POSITION == 1 else "call"
        if response == "call":
            player1["chips"] -= 1
            pot += 1
            print(f"{player1['name']} calls. Pot is now {pot}.")
        elif response == "fold":
            print(f"{player1['name']} folds. {player2['name']} wins the pot of {pot} chips!")
            return
    else:
        print(f"Both players check. Moving to showdown.")

    # Showdown
    print("\n--- Showdown ---")
    winner = determine_winner(player1, player2)
    print(f"The winner is {winner['name']} with card {winner['card']}.")
    winner["chips"] += pot

    # Print chip counts
    print(f"\n--- Chip Counts ---")
    print(f"{player1['name']}: {player1['chips']} chips")
    print(f"{player2['name']}: {player2['chips']} chips")

# Function for user action
def get_user_action(player, allowed_actions):
    while True:
        action = input(f"{player['name']}, choose your action ({allowed_actions}): ").lower()
        if action in allowed_actions.split(" or "):
            return action
        print("Invalid action. Please try again.")

# Function for algorithm action
def get_algorithm_action(player, opponent_action):
    if opponent_action == "check":
        # Algorithm bets with "K", checks with "Q" or "J"
        if player["card"] == "K":
            print(f"{player['name']} bets.")
            return "bet"
        else:
            print(f"{player['name']} checks.")
            return "check"
    elif opponent_action == "bet":
        # Algorithm calls with "K" or "Q", folds with "J"
        if player["card"] in ["K", "Q"]:
            print(f"{player['name']} calls.")
            return "call"
        else:
            print(f"{player['name']} folds.")
            return "fold"

# Function to determine the winner
def determine_winner(player1, player2):
    if CARDS.index(player1["card"]) > CARDS.index(player2["card"]):
        return player1
    else:
        return player2

# Main function to play the game
if __name__ == "__main__":
    print("Welcome to Kuhn Poker!")
    play_kuhn_poker()
