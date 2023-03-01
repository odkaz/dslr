import random



def count_elements(seq) -> dict:
    hist = {}
    for i in seq:
        hist[i] = hist.get(i, 0) + 1
    return hist

def ascii_histogram(seq) -> None:
    counted = count_elements(seq)
    for k in sorted(counted):
        print('{0:5d} {1}'.format(k, '+' * counted[k]))

# main1
# a = (0, 1, 1, 1, 2, 3, 7, 7, 23)
# counted = count_elements(a)
# print(counted)


# main2
# random.seed(1)
# vals = [1, 3, 4, 6, 8, 9, 10]
# freq = (random.randint(5, 15) for _ in vals)
# data = []
# for f, v in zip(freq, vals):
#     data.extend([v] * f)
# ascii_histogram(data)