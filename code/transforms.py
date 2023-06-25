from imports import *#

load_transforms = Compose(
    [
        LoadImageD(keys=["im"], image_only=False),
        EnsureChannelFirstD(keys=["im"]),
        ScaleIntensityRanged(keys=["im"],
            a_min=-200,
            a_max=200,
            b_min=0.0,
            b_max=1.0,
            clip=True,),
        Orientationd(keys=["im"], axcodes="LA"),
        Rotate90d(["im"], k=1, spatial_axes=(0, 1)),
        EnsureTypeD(keys=["im"]),
    ]
)

train_transforms = Compose(
    [
        load_transforms,
    ]
).flatten()

test_transforms = Compose(
    [
        load_transforms,
    ]
).flatten()