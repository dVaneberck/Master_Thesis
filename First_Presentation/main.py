import sys
from cartpole import exec_cartpole
from mountaincar import exec_mountaincar


def main(args):
    if int(args[1]) == 1:
        exec_cartpole()
    elif int(args[1]) == 2:
        exec_mountaincar()
    else:
        print("Value 1 (cartpole) or value 2 (mountaincar) expected.")


if __name__ == '__main__':
    main(sys.argv)
