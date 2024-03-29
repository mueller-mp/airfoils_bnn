'''
script to execute the train.py file several times with different parameters
'''
import click
import subprocess


@click.command()
@click.option("--datadir", default = "../data/")
@click.option("--model_types", default = 'bayesian-unet,bayesian-mars-moon',
              help = 'bayesian-mars-moon or bayesian-unet')
@click.option("--batch_size", default = 64, type = click.INT)
@click.option("--lrg", default = 0.005, type = click.FLOAT)
@click.option("--epochs", default = 40, type = click.INT)
@click.option("--kl_prefs", default = '1,10,100,1000,10000')
@click.option("--expo", default = 3., type = click.FLOAT)
@click.option("--dropouts", default = '0.01,0.05,0.1,0.25')
@click.option("--folder", default = '')
@click.option("--save_model", default = False, type = click.BOOL)
@click.option("--seed", default = 405060, type = click.INT)
def main(datadir, model_types, batch_size, lrg, epochs, kl_prefs, expo, dropouts, folder, save_model, seed, ):
    def from_string_list(string_list, dtype=int):
        try:
            return list(map(dtype, string_list.split(",")))
        except:
            return string_list

    kl_prefs = from_string_list(kl_prefs, dtype = float)
    dropouts = from_string_list(dropouts, dtype = float)
    model_types = from_string_list(model_types, dtype = str)
    common_flags = ["datadir={}".format(datadir), "batch_size={}".format(batch_size), "lrg={}".format(lrg),
        "epochs={}".format(epochs), "folder={}".format(folder), "save_model={}".format(save_model),
        "seed={}".format(seed), "expo={}".format(expo), ]

    for model_type in model_types:
        # # 1. Non-Bayesian
        for dropout in dropouts:
            flags = common_flags + ["model_type={}".format(model_type), 'flipout={}'.format(False),
                'dropout={}'.format(dropout), "kl_pref={}".format(1.), "val_training={}".format(False)]
            flags = " ".join(map(lambda x: "--{}".format(x), flags))
            command = "python train.py {}".format(flags)
            print(command)
            subprocess.call(command, shell = True)

        # # 2. flipout BNN
        # for dropout in [0., 0.01]:
        #     for kl_pref in kl_prefs:
        #         flags = common_flags + [
        #             "model_type={}".format(model_type),
        #             'flipout={}'.format(True),
        #             'dropout={}'.format(dropout),
        #             "kl_pref={}".format(kl_pref),
        #             "val_training={}".format(False)
        #         ]
        #         flags = " ".join(map(lambda x: "--{}".format(x), flags))
        #         command = "python train.py {}".format(flags)
        #         print(command)
        #         subprocess.call(command, shell=True)

        # 3. dropout BNN
        for spatial_dropout in [True, False]:
            for dropout in dropouts:
                flags = common_flags + ["model_type={}".format(model_type), 'flipout={}'.format(False),
                    'dropout={}'.format(dropout), "kl_pref={}".format(1.), "spatial_dropout={}".format(spatial_dropout),
                    "val_training={}".format(False), ]
                flags = " ".join(map(lambda x: "--{}".format(x), flags))
                command = "python train_new.py {}".format(flags)
                print(command)
                subprocess.call(command, shell = True)


if __name__ == "__main__":
    main()
