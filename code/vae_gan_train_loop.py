from imports import *

BCELoss = torch.nn.BCELoss(reduction="sum")
disc_loss = torch.nn.BCELoss()
gen_loss = torch.nn.BCELoss()
real_label = 1
gen_label = 0

def vae_loss(recon_x, x, mu, log_var, beta):
    bce = BCELoss(recon_x, x)
    kld = -0.5 * beta * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    #print(f"bce: {bce}, kld: {kld}")
    return bce + kld


def discriminator_loss(gen_images, real_images, disc_net):
    real = real_images.new_full((real_images.shape[0], 1), real_label)
    gen = gen_images.new_full((gen_images.shape[0], 1), gen_label)

    realloss = disc_loss(disc_net(real_images), real)
    genloss = disc_loss(disc_net(gen_images.detach()), gen)

    #print(f"real: {realloss:.3f}, gen: {genloss:.3f}")
    return (realloss + genloss) / 2



def train(in_shape, max_epochs, latent_size, learning_rate, beta, train_loader, test_loader, device, save_path):
    model = VarAutoEncoder(
        spatial_dims=2,
        in_shape=in_shape,
        out_channels=1,
        latent_size=latent_size,
        channels=(16, 32, 64, 128, 256),
        strides=(1, 2, 2, 2, 2),
        # channels=(16, 32, 64), #change in generate images too
        # strides=(1, 2, 2),
    ).to(device)

    disc_net = Discriminator(
        in_shape = in_shape, 
        channels=(16, 32, 64, 128, 256), 
        strides=(1, 2, 2, 2, 2), 
        kernel_size=3, 
        num_res_units=2, 
        act='PRELU', 
        norm='INSTANCE', 
        dropout=0.25, 
        bias=True, 
        last_act='SIGMOID'
    ).to(device)

    # Create optimiser
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    disc_opt = torch.optim.Adam(disc_net.parameters(), learning_rate)

    avg_train_losses = []
    avg_disc_losses = []
    disc_losses = []
    test_losses = []
    step = 0
    disc_train_interval = 10

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        disc_total_loss =0

        for batch_data in train_loader:
            inputs = batch_data["im"].to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var, _ = model(inputs)

            vloss = vae_loss(recon_batch, inputs, mu, log_var, beta)
            dloss = discriminator_loss(recon_batch, inputs, disc_net)
            loss = vloss * (1-dloss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            disc_losses.append(dloss.item())
            disc_total_loss += dloss.item()

            if dloss.item() > 0.5:
            #if step % disc_train_interval == 0:
                disc_opt.zero_grad()
                dloss = discriminator_loss(recon_batch, inputs, disc_net)
                dloss.backward()
                disc_opt.step()
                
            step += 1

        avg_train_losses.append(epoch_loss / len(train_loader.dataset))
        avg_disc_losses.append(disc_total_loss / len(train_loader.dataset))

        # Test
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_data in test_loader:
                inputs = batch_data["im"].to(device)
                recon, mu, log_var, _ = model(inputs)
                # sum up batch loss
                test_loss += vae_loss(recon, inputs, mu, log_var, beta).item() + discriminator_loss(recon, inputs, disc_net).item()

        test_losses.append(test_loss / len(test_loader.dataset))
        print(f"epoch {epoch + 1}, average train loss: " f"{avg_train_losses[-1]:.4f}, average discriminator loss: " f"{avg_disc_losses[-1]:.4f}, test loss: {test_losses[-1]:.4f}")

        torch.save(model.state_dict(), os.path.join(save_path, "trained_model.pth")) 

    return avg_train_losses, disc_losses, test_losses