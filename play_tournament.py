import sys
import matplotlib.pyplot as plt
import SimWorlds
import TOPP


def main():
    model_file_name = sys.argv[1]

    hex_board = int(sys.argv[2])

    GAMES = 20

    if len(sys.argv) > 3:
        GAMES = int(sys.argv[3])

    sim_world = SimWorlds.HexGame(hex_board)

    topp = TOPP.TOPP()

    topp.run_tournament_distinct_model(model_file_name, GAMES, sim_world)

    return


if __name__ == "__main__":
    main()
