

def get_default_settings(is_3D):
    return {
        'unet_kern_size': 5 if not is_3D else 3,
        'unet_n_first': 32,
        'unet_n_depth': 2,
        'unet_residual': False,
        'train_learning_rate': 0.0004,
        'n2v_perc_pix': 0.198,
        'n2v_neighborhood_radius': 5,
        'denoiseg_alpha': 0.5,
        'single_net_per_channel': True,
        'relative_weights': [1.0, 1.0, 5.0]
    }
