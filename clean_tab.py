import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', required=True)
    parser.add_argument('-o', '--output_path', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input_path) as fp:
        lines = fp.readlines()
    for i in range(len(lines)):
        s = lines[i]
        p = s.find('\t', s.find('\t') + 1) + 1
        lines[i] = s[:p] + s[p:].replace('\t', ' ')
    with open(args.output_path, 'w') as fp:
        fp.writelines(lines)


if __name__ == '__main__':
    main()

