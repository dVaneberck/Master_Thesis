import sys
from old.cartpole import exec_cartpole
# from todo.mountaincar import exec_mountaincar
# from todo.breakout import exec_breakout


def main(args):
    if int(args[1]) == 1:
        exec_cartpole()
    # elif int(args[1]) == 2:
    #     exec_mountaincar()
    # elif int(args[1]) == 3:
    #     exec_breakout()
    else:
        print("Value 1 (cartpole) or value 2 (mountaincar) expected.")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Value 1 (cartpole) or value 2 (mountaincar) expected.")
    else:
        main(sys.argv)
