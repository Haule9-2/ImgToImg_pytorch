import os
import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    # Parse training options
    opt = TrainOptions().parse()  # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    visualizer = Visualizer(opt)  # create a visualizer to display the results

    total_iters = 0  # total number of training iterations
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        model.set_epoch(epoch)  # set the epoch for the model
        epoch_start_time = time.time()  # timer for epoch
        epoch_iter = 0  # number of iterations in this epoch

        for i, data in enumerate(dataset):
            total_iters += opt.batch_size  # increment total number of iterations
            epoch_iter += opt.batch_size  # increment the number of iterations in this epoch

            model.set_input(data)  # unpack data from the dataloader
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            # Display results and save images
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # Print training statistics
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, time.time() - epoch_start_time)
                epoch_start_time = time.time()  # reset timer

        # Save the model after each epoch
        if epoch % opt.save_epoch_freq == 0:
            model.save_networks('latest')  # save latest model
            model.save_networks(epoch)  # save model for current epoch

        # Load latest model if specified for continuing training
        if opt.continue_train:
            model.load_networks('latest')

        # Update learning rate
        model.update_learning_rate()  # update learning rates at the end of the epoch

    # Save the final model
    model.save_networks('final')
