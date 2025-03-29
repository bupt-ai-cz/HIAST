from yacs.config import CfgNode as cn

cfg = cn()
cfg.trainer = None
cfg.work_dir = './'
# ==========================================================================================
# model and loss
# ==========================================================================================
cfg.model = cn()
cfg.model.type = None
cfg.model.is_freeze_bn = True  # should be True after source only training

# segment model
cfg.model.seg_model = cn()
cfg.model.seg_model.type = 'DeepLab_V2'
cfg.model.seg_model.output_dim = 256  # dimensions of embedding feature

# predictor
cfg.model.predictor = cn()

## segment loss
cfg.model.predictor.seg_loss = cn()
cfg.model.predictor.seg_loss.type = 'CE'
cfg.model.predictor.seg_loss.source_weight = 1.0
cfg.model.predictor.seg_loss.target_pseudo_weight = 1.0  # seg loss weight of target pseudo labels

## kld loss (confident region for self training)
cfg.model.predictor.kld_loss = cn()
cfg.model.predictor.kld_loss.weight = 0.1

## entropy loss (all region for adversarial training, ignored region for self training)
cfg.model.predictor.ent_loss = cn()
cfg.model.predictor.ent_loss.weight = 3.0

# discriminator
cfg.model.discriminator = cn()
cfg.model.discriminator.is_enabled = False  # should be True when adversarial training
cfg.model.discriminator.is_entropy_input = False  # use entropy as input of discriminator
cfg.model.discriminator.lr = 1e-4

## discriminator and adversarial loss
cfg.model.discriminator.D_loss = cn()
cfg.model.discriminator.D_loss.type = 'MSE'  # 'MSE' for IAST, 'BCEWithLogits' for AdaptSegNet and AdvEnt
cfg.model.discriminator.D_loss.weight = 1.0
cfg.model.discriminator.D_loss.adv_weight = 0.05

# ==========================================================================================
# dataset
# ==========================================================================================
cfg.dataset = cn()
cfg.dataset.num_classes = 19  # 19 for GTAV to Citysvapes, 9 for Cityscapes to Oxford RobotCar
cfg.dataset.num_workers = 2

# source dataset
cfg.dataset.source = cn()
cfg.dataset.source.type = None  # 'GTAV', 'SYNTHIA', 'Cityscapes'
cfg.dataset.source.json_path = None
cfg.dataset.source.image_dir = None
cfg.dataset.source.aug_type = []  # an list of str, eg. [None] or ['MS', 'SCA']

# target dataset
cfg.dataset.target = cn()
cfg.dataset.target.type = None  # 'Cityscapes', 'Oxford'
cfg.dataset.target.json_path = None
cfg.dataset.target.image_dir = None
cfg.dataset.target.pseudo_dir = None
cfg.dataset.target.aug_type = []  # an list of str, eg. [None] or ['MS', 'SCA']

# validation dataset
cfg.dataset.val = cn()
cfg.dataset.val.type = None  # 'Cityscapes', 'Oxford'
cfg.dataset.val.json_path = None
cfg.dataset.val.image_dir = None
cfg.dataset.val.resize_size = None  # [height, width]

# ==========================================================================================
# generating pseudo labels
# ==========================================================================================
cfg.pseudo_policy = cn()
cfg.pseudo_policy.resume_from = None
cfg.pseudo_policy.batch_size = 2
cfg.pseudo_policy.resize_size = None  # [height, width]
cfg.pseudo_policy.save_dir = None
cfg.pseudo_policy.type = None  # 'IAS', 'CBST', 'CT', 'NT'

# IAS, http://arxiv.org/abs/2008.12197
cfg.pseudo_policy.ias = cn()
cfg.pseudo_policy.ias.alpha = 0.2
cfg.pseudo_policy.ias.beta = 0.9
cfg.pseudo_policy.ias.gamma = 8.0

# CBST, http://openaccess.thecvf.com/content_ECCV_2018/html/Yang_Zou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.html
cfg.pseudo_policy.cbst = cn()
cfg.pseudo_policy.cbst.p = 0.2  # similar with alpha in IAS, controlling the selected pixels of each class
cfg.pseudo_policy.cbst.sample_interval = 4  # the sampling interval for counting the prediction probability of each category in CBST. If all pixels are used, the memory will explode

# constant threshold
cfg.pseudo_policy.ct = cn()
cfg.pseudo_policy.ct.threshold = 0.9

# ==========================================================================================
# training
# ==========================================================================================
cfg.train = cn()
cfg.train.batch_size = 4
cfg.train.lr = 1e-4  # learning rates of backbone, learning rates of other modules please refer to the get_optimizer_params()
cfg.train.optimizer = 'Adam'  # 'SGD', 'Adam'
cfg.train.resume_from = None
cfg.train.apex_opt = 'O1'  # 'O0', 'O1', 'O2', 'O3', https://nvidia.github.io/apex/amp.html#opt-levels
cfg.train.gpu_num = 2
cfg.train.random_seed = 888
cfg.train.port = 6789  # port for distributed training
cfg.train.is_save_all = False  # whether save model when report iteration
cfg.train.is_debug = False

# iteration
cfg.train.total_iter = 10000
cfg.train.iter_report = 100
cfg.train.iter_val = 400

# learning rate scheduler
cfg.train.lr_scheduler = cn()
cfg.train.lr_scheduler.type = 'Cosine'  # 'Cosine', 'Poly'

## poly
cfg.train.lr_scheduler.poly = cn()
cfg.train.lr_scheduler.poly.power = 0.9

# ==========================================================================================
# validate
# ==========================================================================================
cfg.validate = cn()
cfg.validate.resume_from = None
cfg.validate.resize_sizes = []  # multiple scales input, [[height, width], ...]
cfg.validate.is_flip = False
cfg.validate.batch_size = 2
cfg.validate.color_mask_dir_path = None

# ==========================================================================================
# consistency training
# ==========================================================================================
cfg.cst_training = cn()
cfg.cst_training.is_enabled = False

# momentum model
cfg.cst_training.ema_model = cn()
cfg.cst_training.ema_model.iter_update = 1
cfg.cst_training.ema_model.gamma = 0.999

# consistency loss
cfg.cst_training.cst_loss = cn()
cfg.cst_training.cst_loss.type = 'SoftCE'
cfg.cst_training.cst_loss.weight = 1.0
cfg.cst_training.cst_loss.region = 'ignored'  # 'confident', 'ignored', 'all'

# ==========================================================================================
# mutual training
# ==========================================================================================
cfg.mut_training = cn()
cfg.mut_training.is_enabled = False
cfg.mut_training.resume_from = None  # counterpart model resume from
cfg.mut_training.is_strong_input = False  # whether the image input to another network has undergone strong augmentation during mutual learning

# mutual loss
cfg.mut_training.mut_loss = cn()
cfg.mut_training.mut_loss.weight = 0.1
cfg.mut_training.mut_loss.region = 'ignored'

# ==========================================================================================
# data preprocess policy (ClassMix, CutMix, Copy Paste)
# ==========================================================================================
cfg.preprocessor = cn()
cfg.preprocessor.type = None  # 'CopyPaste', 'CutMix', 'ClassMix' only support 'CopyPaste' now

# Copy Paste
cfg.preprocessor.copy_paste = cn()
cfg.preprocessor.copy_paste.mode = 'original'  # 'original', 'consistency', 'multi-scales-prevent-destruction' only support 'original' now
cfg.preprocessor.copy_paste.name = 'normal'

## general setting of copy paste
cfg.preprocessor.copy_paste.selected_num_classes = 14  # how many categories are selected for each image
cfg.preprocessor.copy_paste.gamma = 0.99
