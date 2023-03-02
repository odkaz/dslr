import sys
import util

# main
def main():
    if len(sys.argv) != 2:
        url = './datasets/dataset_train.csv'
    else:
        url = sys.argv[1]

    data = util.read_data(url)
    util.show_data(data)

if __name__ == '__main__':
    main()
