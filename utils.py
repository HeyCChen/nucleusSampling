import os
import csv


def num_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def results_to_file(args, test_acc):

    if not os.path.exists('./results/{}'.format(args.dataset)):
        print("="*20)
        print("Create Results File !!!")

        os.makedirs('./results/{}'.format(args.dataset))

    filename = "./results/{}/{}.csv".format(
        args.dataset, args.sampling_method,)

    headerList = ["Method", "Seed-id",
                  "::::::::",
                  "test_acc"]

    with open(filename, "a+") as f:

        # reader = csv.reader(f)
        # row1 = next(reader)
        f.seek(0)
        header = f.read(6)
        if header != "Method":
            dw = csv.DictWriter(f, delimiter=',',
                                fieldnames=headerList)
            dw.writeheader()

        line = "{}, {}, :::::::::, {:.4f}\n".format(
            args.sampling_method, args.seed,
            test_acc
        )
        f.write(line)
