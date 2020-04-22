from HexGame import HexGame
from ActorNet import LoadedNet


def test_illegal_action():
    net = LoadedNet("After_3_340")

    for i in range(50):
        game = HexGame()
        while not game.is_done():
            state = game.get_state()
            pre_screen = net.get_propabilities(state)
            after_screen = game.get_possible_and_normalized_possibilities(pre_screen)
            action = game.get_action(1,after_screen)

            if action not in game.get_possible_actions():
                print(f"Action illegaly chosen: {action}")
                print("Action: ".ljust(10) + "Value:".ljust(10) + "Pre screen".ljust(20) + "After screen:".ljust(20))
                for i in range(36):
                    current_action = (i//6,i%6)
                    print(str(current_action).ljust(10) + str(state[i+1]).ljust(10) + str(pre_screen[i]).ljust(20) + str(after_screen[i]).ljust(20))
                game.board.show_graph(debug=True, pause=100)
                
            game.perform_action(action)