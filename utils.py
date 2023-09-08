import os
import csv
import numpy as np


def num_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def results_to_file(args, test_acc):

    if not os.path.exists('./results/{}'.format(args.dataset)):
        print("="*20)
        print("Create Results File !!!")

        os.makedirs('./results/{}'.format(args.dataset))

    filename = "./results/{}/{}{}.csv".format(
        args.dataset, args.sampling_method, args.pooling_ratio)

    headerList = ["Method", "Seed-id", "Pooling-ratio",
                  "test_acc"]

    with open(filename, "a+") as f:
        f.seek(0)
        header = f.read(6)
        if header != "Method":
            dw = csv.DictWriter(f, delimiter=',',
                                fieldnames=headerList)
            dw.writeheader()

        line = "{},{},{},{:.4f}\n".format(
            args.sampling_method, args.seed, args.pooling_ratio,
            test_acc
        )
        f.write(line)


def results_to_compare(args):
    filename = "./results/{}/{}{}.csv".format(
        args.dataset, args.sampling_method, args.pooling_ratio)
    with open(filename) as csv_file:
        row = csv.reader(csv_file, delimiter=',')
        next(row)
        arr = []
        for r in row:
            arr.append(float(r[3]))

    mean = np.mean(arr)
    var = np.var(arr)

    if not os.path.exists('./results/{}'.format(args.dataset)):
        os.makedirs('./results/{}'.format(args.dataset))

    resultname = "./results/{}/compare.csv".format(args.dataset)
    headerList = ["Method", "Pooling-ratio", "Mean", "Var"]

    with open(resultname, "a+") as f:
        f.seek(0)
        header = f.read(6)
        if header != "Method":
            dw = csv.DictWriter(f, delimiter=',',
                                fieldnames=headerList)
            dw.writeheader()

        line = "{},{},{:.4f},{:.6f}\n".format(
            args.sampling_method, args.pooling_ratio,
            mean, var
        )
        f.write(line)
