import time
import os
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

def save_networks(model, save_suffix):
    """Saves the model and optimizer states."""
    save_path = os.path.join(model.checkpoints_dir, f'{model.name}_{save_suffix}.pth')
    
    state_dict = {
        'model': model.netG.state_dict(),          # Save generator model state
        'optimizer': model.optimizer.state_dict(),  # Save optimizer state
        'epoch': model.epoch,                       # Save current epoch
        'iter': model.iteration                     # Save current iteration
    }
    
    torch.save(state_dict, save_path)
    print(f'Saved model checkpoints to {save_path}')

def load_networks(model, checkpoint_path):
    """Loads the model and optimizer states."""
    print("loading")
    checkpoint = torch.load(checkpoint_path)

    # Load generator model weights
    model.netG.load_state_dict(checkpoint['model'])  
    # Load optimizer state
    model.optimizer.load_state_dict(checkpoint['optimizer'])  
    # Restore current epoch
    model.epoch = checkpoint['epoch']                      
    # Restore current iteration
    model.iteration = checkpoint['iter']                   

    # Fixed print statement to show loaded epoch and iteration correctly
    print(f'Loaded checkpoint from {checkpoint_path} with epoch: {model.epoch}, iteration: {model.iteration}')

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print("================================================================")
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    # Load previous checkpoints if continue training
    if opt.continue_train:
        checkpoint_path = os.path.join(opt.checkpoints_dir, f'latest_{opt.name}.pth')
        if os.path.exists(checkpoint_path):
            load_networks(model, checkpoint_path)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in the current epoch
        visualizer.reset()              # reset the visualizer
        model.update_learning_rate()    # update learning rates at the start of every epoch

        for i, data in enumerate(dataset):  
            iter_start_time = time.time()  
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   
                print('Saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                save_networks(model, save_suffix)  # Save networks

            iter_data_time = time.time()
        
        if epoch % opt.save_epoch_freq == 0:              
            print('Saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            save_networks(model, 'latest')  # Save latest model
            save_networks(model, epoch)      # Save model by epoch

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
