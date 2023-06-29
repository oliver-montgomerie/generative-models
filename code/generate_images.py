from imports import *

## viewing
def generate_images(num_ims, img_shape, latent_size, device, save_path):
    model = VarAutoEncoder(
        spatial_dims=2,
        in_shape=img_shape,
        out_channels=1,
        latent_size=latent_size,
        channels=(16, 32, 64, 128, 256),
        strides=(1, 2, 2, 2, 2),
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(save_path, "trained_model.pth")))
    model.eval()

    dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))

    with torch.no_grad():
        for i in range(num_ims):
            sample = torch.zeros(1, latent_size).to(device)
            for s in range(sample.shape[1]):
                sample[0,s] = dist.icdf((torch.rand(1)*0.9) + 0.05)

            o = model.decode_forward(sample)
            o = o.detach().cpu().numpy().reshape(img_shape[1:])
            plt.figure("generated")
            plt.imshow(o, cmap="gray")
            plt.savefig(os.path.join(save_path, "img-"+str(i)), bbox_inches='tight')
            plt.close()   

    # with torch.no_grad():
    #     for z in range(latent_size - 1):
    #         for z in range(latent_size - 1):
    #             for y, j in enumerate(torch.linspace(0.05, 0.95, num_ims)):
    #                 for i in torch.linspace(0.05, 0.95, num_ims):
    #                     sample = torch.zeros(1, latent_size).to(device)
    #                     sample[0, z] = dist.icdf(j)
    #                     sample[0, z + 1] = dist.icdf(i)
    #                     o = model.decode_forward(sample)
    #                     o = o.detach().cpu().numpy().reshape(img_shape[1:])
    #                     plt.figure("generated")
    #                     plt.imshow(o, cmap="gray")
    #                     plt.savefig(os.path.join(save_path, "img-"+str(z)+"-"+str(y)), bbox_inches='tight')
    #                     plt.close()   




# im_dir = "/home/omo23/Documents/sliced-data/Images"
# all_filenames = [os.path.join(im_dir, filename) for filename in os.listdir(im_dir)]
# test_frac = 0.2
# num_test = int(len(all_filenames) * test_frac) #2
# test_datadict = [{"im": fname} for fname in all_filenames[-num_test:]]
# from transforms import test_transforms
# im_shape = test_transforms(test_datadict[0])["im"].shape
# latent_size = 5
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# save_path = "/home/omo23/Documents/generative-models/VAE-generated/latent-5-epochs-20"

# generate_images(num_ims=10,
#                 img_shape=im_shape,
#                 latent_size=latent_size,
#                 device=device,
#                 save_path=save_path)