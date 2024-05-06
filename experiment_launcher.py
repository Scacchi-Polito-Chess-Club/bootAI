import datetime

import tqdm

import games_from_dataset as gd
import torch.optim
import torch.utils.data as data
import tqdm
from utils.utils_model import *
import wandb
import time


@torch.no_grad()
def test(model: nn.Module, test_data: data.DataLoader, config, logger):
    device = config['setup_args']['device']
    loss_func = get_loss_func(config['exp_args']['loss'])
    corrects = 0
    totals = 0
    tot_loss = 0.0
    for (b1, b2), move_gt in test_data:
        b1, b2, move_gt = b1.to(device).float(), b2.to(device).float(), move_gt.to(device).float()
        move = model(b1, b2)
        predicted_move = torch.argmax(move, axis=-1)
        gt = torch.argmax(move_gt, axis=-1)
        corrects += torch.sum(predicted_move == gt)
        totals += predicted_move.shape[0]
        tot_loss += loss_func(move, move_gt).item()
    accuracy = (corrects/totals).detach().cpu().numpy()*100
    avg_loss = tot_loss / len(test_data)
    if logger is not None:
        logger.info(f"\nEval accuracy {accuracy:.2f}%, Eval avg loss {avg_loss:.5f}")
        wandb.log({"Eval accuracy": accuracy, "Eval avg loss": avg_loss})
    return accuracy, avg_loss


def train(model: nn.Module, train_data: data.DataLoader, val_data: data.DataLoader, config, logger):
    # check if there is a model to be resumed
    if config['setup_args']['resume']:
        try:
            pt_file = config['setup_args']['resume_path']
            checkpoint = torch.load(pt_file)
        except FileNotFoundError as e1:
            print(f"File path incorrect or null: {config['setup_args']['resume_path']}")
            raise e1
        except TypeError as e2:
            print(f"File path is not a string: {type(config['setup_args']['resume_path'])}")
            raise e2
        print('Resume Training')
        device = config['setup_args']['device']
        model = model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim = get_optimizer(model, config['exp_args']['optimizer'].lower(), config['exp_args']['lr'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        sched = get_scheduler(optim, type=config['exp_args']['scheduler'])
        sched.load_state_dict(checkpoint['scheduler_state_dict'])
        loss_func = get_loss_func(config['exp_args']['loss'])
        init_epoch = checkpoint['epoch'] + 1

    else:
        print('Start Training')
        pt_file = datetime.datetime.now().strftime("./models/%Y-%m-%d-%H-%M") + ".pt"
        device = config['setup_args']['device']
        model = model.to(device)
        optim = get_optimizer(model, config['exp_args']['optimizer'].lower(), config['exp_args']['lr'])
        sched = get_scheduler(optim, type=config['exp_args']['scheduler'])
        loss_func = get_loss_func(config['exp_args']['loss'])
        init_epoch = 0

    if logger is not None:
        logger.info('\nStart Training')
        logger.info(config['setup_args']['device'])
        logger.info(config['exp_args'])
        logger.info(config['encoder'])
        logger.info(config['decoder'])

    for epoch in range(init_epoch, config['exp_args']['epoch']):
        t = time.time()
        tot_loss = 0.0
        corrects = 0
        totals = 0
        loss = torch.tensor(0.0)
        # TODO: handling repetition of games if data is resumed by checkpoint
        data_iterator = tqdm.tqdm(train_data)
        for (b1, b2), action_gt in data_iterator:
            data_iterator.set_description(f'Training epoch {epoch}, training loss {loss.item():5f}')
            b1, b2 = b1.to(device).float(), b2.to(device).float()
            action_gt = action_gt.to(device).float()
            optim.zero_grad()
            action = model(b1, b2)
            loss = loss_func(action, action_gt)
            loss.backward()
            optim.step()
            tot_loss += loss.item()
            predicted_move = torch.argmax(action, axis=-1)
            gt = torch.argmax(action_gt, axis=-1)
            corrects += torch.sum(predicted_move == gt)
            totals += predicted_move.shape[0]
        sched.step()
        t = time.time() - t
        accuracy = (corrects / totals).detach().cpu().numpy()*100
        avg_loss = tot_loss / len(train_data)
        if logger is not None:
            logger.info(f"\nEpoch {epoch}: Train avg loss  {loss:.5f}, "
                        f"Train accuracy {accuracy:.2f}%, Time required: {t:.2f}s")
            wandb.log({"Epoch": epoch, "Train avg loss":  avg_loss,
                       "Train accuracy": accuracy, "Time": t})
        if epoch % config['exp_args']['eval_step'] == 0:
            eval_stats, _ = test(model, val_data, config, logger)
            print(f"Eval accuracy {eval_stats}")
            # checkpoint and saving of model parameters, optimizer, scheduler
            f = open(pt_file, "w")
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'scheduler_state_dict': sched.state_dict(),
                        'loss': tot_loss,
                        }, pt_file)
            f.close()
        if epoch % 5 == 0:
            torch.save(model.state_dict(), "/home/ai-team/botAI/model_epoch_{}.pt".format(epoch))
            torch.save(optim.state_dict(), "/home/ai-team/botAI/optimizer_epoch_{}.pt".format(epoch))
    return model


