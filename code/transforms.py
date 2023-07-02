from imports import *

class flip_if_liver_on_right(MapTransform):
    #for viewing 2d slices
    def __init__(self, keys, label_key):
        self.keys = keys
        self.label_key = label_key

    def __call__(self, data):
        #get location of label == 1
        # if avg idx > width /2 then call flip
        d = dict(data)
        im = d['image']
        lbl = d[self.label_key]
        idx = np.argwhere(lbl>0) #location of liver
        mid_liver = np.mean(idx, axis = 0)

        if mid_liver[2] > lbl.shape[2]/2:
            lbl = torch.flip(lbl, [2])
            im = torch.flip(im, [2])
        d[self.label_key] = lbl
        d['image'] = im
        return d
    

load_slice_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200,
            a_max=200,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Orientationd(keys=["image", "label"], axcodes="LA"),
        Rotate90d(["image", "label"], k=1, spatial_axes=(0, 1)),
        flip_if_liver_on_right(keys=["image", "label"], label_key="label"),
        Spacingd(keys=["image", "label"], pixdim=(0.793, 0.793), mode=("bilinear", "nearest")),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size = [560,560]),
    ]
)


deform = Rand2DElasticd(
    keys = ["im"],
    prob=0.5,
    spacing=(55, 55),
    magnitude_range=(-1.1,1.1),
    rotate_range=(np.pi / 20,),
    shear_range= (-0.05,0.05),
    translate_range=(-5, 5),
    scale_range=(-0.1, 0.1),
    padding_mode="zeros",
)


load_tumor_transforms = Compose(
    [
        LoadImaged(keys=["im"], image_only=False),
        EnsureChannelFirstd(keys=["im"]),
        ScaleIntensityRanged(keys=["im"],
            a_min=-200,
            a_max=200,
            b_min=0.0,
            b_max=1.0,
            clip=True,),
        Orientationd(keys=["im"], axcodes="LA"),
        Spacingd(keys=["im"], pixdim=(0.793, 0.793), mode=("bilinear")),
        deform,
        ResizeWithPadOrCropd(keys=["im"], spatial_size = [256,256]),
        EnsureTyped(keys=["im"]),
    ]
).flatten() 

