from imports import *

def generate_a_tumor(model, dist, tumor_shape, device):
    with torch.no_grad():
        sample = torch.zeros(1, latent_size).to(device)
        for s in range(sample.shape[1]):
            #todo: maybe This should be weighted towards the edges for unusual looking tumors?
            sample[0,s] = dist.icdf((torch.rand(1)*0.9) + 0.05)

        o = model.decode_forward(sample) #get output from latent sample
        o = o.detach().cpu().numpy().reshape(tumor_shape)
        mask = np.zeros(o.shape)    #create mask from thresholding tumor
        thresh = (np.max(o) + np.min(o))/2
        mask[o > thresh] = 1
        #clip the tumor to the mask size
        o[mask != 1] = 0

        return o[0,:,:], mask[0,:,:]


## viewing
def generate_slices(model, tumor_shape, device):
    img_save_path = "/home/omo23/Documents/generated-data/VAE-GAN/Images"
    lbl_save_path = "/home/omo23/Documents/generated-data/VAE-GAN/Labels"
    
    #Data loading
    data_dir = "/home/omo23/Documents/sliced-data"
    all_images = sorted(glob.glob(os.path.join(data_dir, "Images", "*.nii")))
    all_labels = sorted(glob.glob(os.path.join(data_dir, "Labels", "*.nii")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(all_images, all_labels)]
    # filter FOR slices with small tumor area or no tumor
    data_dicts = [item for item in data_dicts if file_tumor_size(item) == 0] # min_tumor_size]

    gen_files = []
    for d in data_dicts:
        d_num = d['image']
        d_num = d_num[d_num.rfind("/")+1:d_num.rfind("-")] 

        if d_num not in test_files_nums + val_files_nums:
            gen_files.append(d)

    t_spacing = Spacing(pixdim=(0.793, 0.793), mode=("bilinear"))

    for file in gen_files:
        img_path = file['image']
        lbl_path = file['label']
        fname = img_path[img_path.rfind("/")+1:img_path.rfind(".")] 

        nib_img = nib.load(img_path)
        nib_lbl = nib.load(lbl_path)

        img = np.array(nib_img.dataobj)
        lbl = np.array(nib_lbl.dataobj)

        dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))

        liver_pix = np.argwhere(lbl == 1)

        max_attempts = 20
        good_placement = False
        for attempt in range(max_attempts):
            tumor_img, tumor_lbl = generate_a_tumor(model, dist, tumor_shape, device)
            # make sure its large enough
            tumor_size = np.sum(tumor_lbl == 1) * nib_lbl.header['pixdim'][1] * nib_lbl.header['pixdim'][2]
            tumor_size = int(tumor_size)
            if tumor_size < min_tumor_size:
                continue

            #resample to images resolution (from (0.793, 0.793))
            # todo: this ^ use the t_spacing above?

            #rescale back to hounsfield(?)
            tumor_img = (tumor_img * 400) - 200

            # pad to slice size
            pad_size = img.shape - np.array([tumor_img.shape[0],tumor_img.shape[1]])
            tumor_img = np.pad(tumor_img, [(pad_size[0], 0), (pad_size[1], 0)], mode='constant', constant_values=np.min(tumor_img))

            pad_size = lbl.shape - np.array([tumor_lbl.shape[0],tumor_lbl.shape[1]])
            tumor_lbl = np.pad(tumor_lbl, [(pad_size[0], 0), (pad_size[1], 0)], mode='constant', constant_values=0)
            
            
            #choose location to insert into liver
            tumor_pix = np.argwhere(tumor_lbl == 1)
            tumor_centre = np.mean(tumor_pix, axis = 0,dtype=int)

            location = random.choice(liver_pix)

            #shift the tumor so that the centre = location.
            tumor_lbl = np.roll(tumor_lbl, location[0] - tumor_centre[0], axis=0)
            tumor_lbl = np.roll(tumor_lbl, location[1] - tumor_centre[1], axis=1)
            tumor_img = np.roll(tumor_img, location[0] - tumor_centre[0], axis=0)
            tumor_img = np.roll(tumor_img, location[1] - tumor_centre[1], axis=1)

            tumor_lbl[tumor_lbl==1] = 3
            # add tumor label to img label. if there are lots of 3's then it means outside liver
            # if lots of 4's then it was inside the liver
            # 5 means on- top of another tumor, which is also ok
            gen_lbl = lbl + tumor_lbl

            # todo: check if this returns the right length or *2 because 2d
            not_liver = len(np.argwhere(gen_lbl == 3))
            in_liver = len(np.argwhere(gen_lbl > 3))

            #print(f"in_liver ratio: {in_liver / (in_liver + not_liver)}")
            if in_liver / (in_liver + not_liver) > 0.95:
                good_placement = True
                break #good

            # Todo: add a probability for repeating the process for adding multiple tumors.
            # look at ratio in other slices and use that??


        if good_placement == False:
            continue
            
        # merge images
        sobel_h = ndimage.sobel(tumor_lbl, 0)  # horizontal gradient
        sobel_v = ndimage.sobel(tumor_lbl, 1)  # vertical gradient
        tumor_edges = np.sqrt(sobel_h**2 + sobel_v**2)
        tumor_edges = tumor_edges / np.max(tumor_edges)
        #tumor_edges = ndimage.gaussian_filter(tumor_edges, sigma = 0.25)

        edge_locations = np.argwhere(tumor_edges > 0.5)
        lbl_locations = np.argwhere(tumor_lbl >= 3)
        dists = cdist(lbl_locations, edge_locations).min(axis=1)
        distmap = np.zeros(tumor_lbl.shape)
        distmap[tumor_lbl >= 3] = dists
        distmap = distmap / (2*np.max(distmap))
        distmap[distmap>0] = distmap[distmap>0] + 0.5

        plt.figure("Processing", (18, 6))
        plt.subplot(2,2,1)
        plt.imshow(tumor_edges, cmap="gray")
        plt.title("tumor edge")
        
        gen_img = np.copy(img)
        gen_img[tumor_lbl >= 3] = tumor_img[tumor_lbl >= 3] #(0.8*tumor_img[tumor_lbl >= 3]) + (0.2*gen_img[tumor_lbl >= 3])

        plt.subplot(2,2,2)
        plt.imshow(gen_img, cmap="gray")
        plt.title("implanted tumor")

        blurred_img = ndimage.gaussian_filter(np.copy(gen_img), sigma = 0.5)
        #blurred_img = ndimage.gaussian_filter(blurred_img, sigma = 0.75)
        blurred_img = ndimage.gaussian_filter(blurred_img, sigma = 1)

        plt.subplot(2,2,3)
        plt.imshow(distmap, cmap="gray")
        plt.title("distance map")

        #gen_img[tumor_lbl >= 3] = ((distmap*tumor_img)[tumor_lbl >= 3]) + (((1-distmap)*img)[tumor_lbl >= 3])

        plt.subplot(2,2,4)
        plt.imshow(gen_img, cmap="gray")
        plt.title("distance map weighted")

        #plt.show()

        plt.figure("New tumor", (18, 6))
        plt.subplot(2,3,1)
        plt.imshow(img, cmap="gray")
        plt.title("Original")
        plt.axis('off')
        plt.subplot(2,3,4)
        plt.imshow(lbl, vmin=0, vmax=5)
        plt.axis('off')

        plt.subplot(2,3,2)
        plt.imshow(tumor_img, cmap="gray")
        plt.axis('off')
        plt.title("Generated tumor")
        plt.subplot(2,3,5)
        plt.imshow(tumor_lbl, vmin=0, vmax=5)
        plt.axis('off')

        plt.subplot(2,3,3)
        plt.imshow(gen_img, cmap="gray")
        plt.title("Implanted tumor")
        plt.axis('off')
        plt.subplot(2,3,6)
        plt.imshow(gen_lbl, vmin=0, vmax=5)
        plt.axis('off')

        #plt.show()

        gen_lbl[gen_lbl >= 3] = 2 
        ni_img = nib.Nifti1Image(gen_img, nib_img.affine) #, img.header (?)
        ni_lbl = nib.Nifti1Image(gen_lbl, nib_lbl.affine, dtype='<i2')

        nib.save(ni_img, os.path.join(img_save_path, fname + "_" + str(tumor_size) + ".nii"))
        nib.save(ni_lbl, os.path.join(lbl_save_path, fname + "_" + str(tumor_size) + ".nii"))


       
    


tumor_shape = [1,256,256]
latent_size = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_path = "/home/omo23/Documents/generative-models/VAE-GAN-models/latent-10-epochs-30"

model = VarAutoEncoder(
    spatial_dims=2,
    in_shape=tumor_shape,
    out_channels=1,
    latent_size=latent_size,
    channels=(16, 32, 64, 128, 256),
    strides=(1, 2, 2, 2, 2),
).to(device)
model.load_state_dict(torch.load(os.path.join(load_path, "trained_model.pth")))
model.eval()

generate_slices(model = model,
                tumor_shape=tumor_shape,
                device=device,)