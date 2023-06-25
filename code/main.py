from imports import *

# Get image original and its degraded versions
def get_single_im(ds):
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=10, shuffle=True)
    itera = iter(loader)
    return next(itera)


def plot_ims(ims, shape=None, figsize=(10, 10), titles=None):
    shape = (1, len(ims)) if shape is None else shape
    plt.subplots(*shape, figsize=figsize)
    for i, im in enumerate(ims):
        plt.subplot(*shape, i + 1)
        im = plt.imread(im) if isinstance(im, str) else torch.squeeze(im)
        plt.imshow(im, cmap="gray")
        if titles is not None:
            plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### data
    im_dir = "C:/Users/olive/OneDrive/Desktop/Liver Files/sliced-data/Images/"
    all_filenames = [os.path.join(im_dir, filename) for filename in os.listdir(im_dir)]
    test_frac = 0.2
    num_test = int(len(all_filenames) * test_frac)
    num_train = len(all_filenames) - num_test
    train_datadict = [{"im": fname} for fname in all_filenames[:num_train]]
    test_datadict = [{"im": fname} for fname in all_filenames[-num_test:]]
    print(f"total number of images: {len(all_filenames)}")
    print(f"number of images for training: {len(train_datadict)}")
    print(f"number of images for testing: {len(test_datadict)}")

    ### dataset and loader

    batch_size = 300
    num_workers = 10

    from transforms import train_transforms, test_transforms

    train_ds = CacheDataset(train_datadict, train_transforms, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_ds = CacheDataset(test_datadict, test_transforms, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    from train_loop import train

    max_epochs = 50
    training_types = ["orig", "gaus", "s&p"]
    models = []
    epoch_losses = []
    for training_type in training_types:
        model, epoch_loss = train(training_type, max_epochs=max_epochs, learning_rate=1e-3, train_loader=train_loader ,device=device)
        models.append(model)
        epoch_losses.append(epoch_loss)


    plt.figure()
    plt.title("Epoch Average Loss")
    plt.xlabel("epoch")
    for y, label in zip(epoch_losses, training_types):
        x = list(range(1, len(y) + 1))
        (line,) = plt.plot(x, y)
        line.set_label(label)
    plt.legend()
    plt.show()

    data = get_single_im(test_ds)

    recons = []
    for model, training_type in zip(models, training_types):
        im = data[training_type]
        recon = model(im.to(device)).detach().cpu()
        recons.append(recon)

    plot_ims(
        [data["orig"], data["gaus"], data["s&p"]] + recons,
        titles=["orig", "Gaussian", "S&P"] + ["recon w/\n" + x for x in training_types],
        shape=(2, len(training_types)),
    )