
config = {

    # network settings
    'nb_conv_blocks': 2,
    'conv_block_type': 'normal',
    'nb_filters': 64,
    'filter_width': 3,
    'nb_units_lstm': 128,
    'nb_layers_lstm': 1,
    'drop_prob': 0.5,
    'error_margins': 5,
    # training settings
    'epochs': 100,
    'batch_size': 10,
    'loss': 'cross_entropy',
    'weighted': False,          #Always false
    'weights_init': 'xavier_uniform',
    'optimizer': 'adam',
    'lr': 1e-4,
    'weight_decay': 1e-6,
    'shuffling': True,
    'valid_type': 'validNotSimplyRegression', #split   #trainValidSimply #other
    'DL_mode': 'regression', # 'regression, 'classification'
    ### UP FROM HERE YOU SHOULD RATHER NOT CHANGE THESE ####
    'window_type': 'window1', # check the type of window in the main regression script
    'valid_epoch':'best',
    'no_lstm': False,
    'batch_norm': False,
    'dilation': 1,
    'pooling': False,
    'pool_type': 'max',
    'pool_kernel_width': 2,
    'reduce_layer': False,
    'reduce_layer_output': 10,
    'nb_classes': 7,
    'seed': 2, #was seed 1
    'gpu': 'cuda:0',
    'verbose': False,
    'print_freq': 10,
    'save_gradient_plot': False,
    'print_counts': False,
    'adj_lr': False,    #no change
    'adj_lr_patience': 5,
    'early_stopping': False,
    'es_patience': 5,
    'save_test_preds': False
}

