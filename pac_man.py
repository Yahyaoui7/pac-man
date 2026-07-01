import sys
from src.parsing import Parser


def main() -> int:
    if len(sys.argv) != 2:
        print("Command should be: python3 pac_man.py config.json")
        return 1

    parser = Parser(sys.argv[1])
    config = parser.parser_all()

    print(config)  # only for testing now
    return 0


if __name__ == "__main__":
    sys.exit(main())
