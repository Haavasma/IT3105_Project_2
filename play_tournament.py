import sys
import matplotlib.pyplot as plt
import SimWorlds
import TOPP


def main():
    model_file_name = sys.argv[1]

    hex_board = int(sys.argv[2])

    sim_world = SimWorlds.HexGame(hex_board)

    topp = TOPP.TOPP()

    topp.run_tournament_distinct_model(model_file_name, 20, sim_world)

    return


if __name__ == "__main__":
    main()
