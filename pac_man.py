import sys
from src.parsing import Parser


def main():

    if len(sys.argv) != 2:
        print("commend run should be like python3 pac_man.py config.json")
        return 1

    parser = Parser(sys.argv[1])
    # try:
    parser.parser_all()
    # except Parser_error as e:




if __name__ == "__main__":
    main()
