from myPackage.util import ceil_sqr


def print_ceil_sqr():
    x = float(input('Insert number:'))
    print(f'Number:{x}, whose ceil squared is {ceil_sqr(x)} ')


if __name__ == '__main__':
    print_ceil_sqr()
