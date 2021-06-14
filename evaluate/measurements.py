
def precision(tp, fp):
    return tp/(tp + fp)


def recall(tp, fn):
    return tp/(tp + fn)


def f1(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 2*p*r/(p+r)


def evaluate(tp, fp, fn):
    print("Precision: ", "%.3f" % precision(tp, fp))
    print("Recall: ", "%.3f" % recall(tp, fn))
    print("F1: ", "%.3f" % f1(tp, fp, fn))


evaluate(15, 0, 2)
