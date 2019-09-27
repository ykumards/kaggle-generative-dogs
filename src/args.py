from pathlib import Path
from argparse import Namespace

args = Namespace(
    seed=123,
    disable_cuda=False,
    device=None,
    debug=False,
    num_workers=4,
    # use this for exp name
    CKPT_PREFIX = 'sagan_hinge_loss',
    exp_name = None,

    ########## Ignite Stuff ###################
    PRINT_FREQ = 600,
    FAKE_IMG_FNAME = 'fake_sample_epoch_{:04d}.png',
    REAL_IMG_FNAME = 'real_sample_epoch_{:04d}.png',
    LOGS_FNAME = 'logs.tsv',
    PLOT_FNAME = 'plot.svg',
    SAMPLES_FNAME = 'samples.svg',
    output_dir = '../output_dir/',
    alpha = 0.98, # smoothing constant for exponential moving average
    
    ######### Dataset Related #################
    shuffle=True,
    datapath=Path('../input/all-dogs/'),
    root_images=Path("../input/all-dogs"),
    root_annots=Path("../input/Annotation/"),
    
    ######### Training Params ######################
    num_epochs=500,
    lrD=4e-4,
    lrG=1e-4,
    beta1 = 0., #SAGAN params (wuuuud?)
    beta2 = 0.9,
    batch_size=64,
    weight_decay=0.001,
    log_interval=100,
    num_disc_update=1,

    ######### Model Params #########################
    num_classes=120,
    image_size=64,
    in_channels=3,
    num_feature_maps_gen=64,
    num_feature_maps_disc=64,
    latent_dim=128,
    input_dim=64*64,
    hidden_size=400,    
    dropout_p=0.2,
)
args.exp_name = args.CKPT_PREFIX + f"_{args.num_epochs}ep_{args.batch_size}bs_{args.num_disc_update}ndisc"
print("="*100)
print(args.exp_name)
print("="*100)



