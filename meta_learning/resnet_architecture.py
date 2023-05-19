resnet34 = [
    ("conv7", {"in_channels": 3, "out_channels": 64}),
    ("max_pool", {"kernel_size": 3, "stride": 2, "padding": 1}),

    ("resblock", {"channels": 64, "repeat": 3}),
    ("down_conv", {"in_channels": 64, "out_channels": 128}),

    ("resblock", {"channels": 128, "repeat": 4}),
    ("down_conv", {"in_channels": 128, "out_channels": 256}),

    ("resblock", {"channels": 256, "repeat": 6}),
    ("down_conv", {"in_channels": 256, "out_channels": 512}),

    ("resblock", {"channels": 512, "repeat": 3}),

    ("adaptive_max_pool", {"output_size": (1, 1, 1)}),
    ("flatten",),
    ("FC", {"in": 512, "out": 3}),
]
