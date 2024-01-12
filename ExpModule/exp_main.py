import os
import time
import math
import random

import torch
import sklearn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
# import matplotlib.animation as animation

from ExpModule._exp_basic import ExpBasic
from utils.dataloader_DL import ExpDataset
from utils.tools import getModelSize, load_model, save_model, EMA


def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    sklearn.utils.check_random_state(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class ExpMain(ExpBasic):
    def __init__(self, args):
        setup_seed(args.seed) if args.seed is not None else setup_seed(2023)
        super(ExpMain, self).__init__(args)
        self.seq_len, self.pred_len = self.args.seq_len, self.args.pred_len
        self.in_dim, self.out_dim = self.args.in_dim, self.args.out_dim

        self.dataset = ExpDataset(self.args.data_name,
                                  self.seq_len, self.pred_len,
                                  self.in_dim, self.out_dim, self.args.normalize)

        # 将路径转换成绝对路径，如果不存在则创建 | Convert the path to an absolute path, if it does not exist, create it
        path = os.getcwd()
        dirname = path[:path.rfind("\\Battery_RUL_Benchmark")] + "/Battery_RUL_Benchmark/"
        for key in self.args.keys():
            if 'path' in key:
                self.args[key] = dirname + self.args[key]
                if not os.path.exists(self.args[key]) and path is not None:
                    os.makedirs(self.args[key])

    def _build_model(self):
        # 根据模型名字，导入模型
        # 如果已经存在最优模型，就加载最优模型
        model_dict = {
            'MLP': 'from models.MLP import Model',
            'GRU': 'from models.GRU import Model',
            'LSTM': 'from models.LSTM import Model',
            'ProgressiveDecompMLP': 'from models.PDMLP_auto import Model',
            'DLinear': 'from models.DLinear import Model',
            'Transformer': 'from models.Transformer import Model',

        }
        exec(model_dict[self.args.model_name])
        model = eval("Model(self.args)")
        model.to(self.args.device) if self.args.use_cuda else model
        # init_params(self.model)
        return model

    @torch.no_grad()
    def vali(self, data_loader, loss_fn, ema):
        loss = torch.tensor(0., device=self.device)
        count = torch.tensor(1e-5, device=self.device)

        # switch to evaluation mode
        self.model.eval()
        ema.apply_shadow() if ema is not None else None
        for batch in data_loader:
            if self.device == torch.device('cpu'):
                x, y = batch
                out = self.model(x)
                tmp_loss = loss_fn(out, y)
                if torch.isnan(tmp_loss).int().sum() == 0:
                    count += 1
                    loss += tmp_loss

            else:
                x, y = [x.half().cuda(non_blocking=True) for x in batch]
                with torch.cuda.amp.autocast():
                    out = self.model(x)
                    tmp_loss = loss_fn(out, y)
                    if torch.isnan(tmp_loss).int().sum() == 0:
                        count += 1
                        loss += tmp_loss
        ema.restore() if ema is not None else None
        loss_val = loss.item() / count.item()
        return loss_val

    @torch.no_grad()
    def finetune_vali(self, data_loader, loss_fn, ema):
        loss = torch.tensor(0., device=self.device)
        count = torch.tensor(1e-5, device=self.device)

        # switch to evaluation mode
        self.model.eval()
        ema.apply_shadow() if ema is not None else None
        for batch in data_loader:
            if self.device == torch.device('cpu'):
                x, y1, y2 = batch
                out = self.model(x)
                loss += loss_fn(out, y1)
                out = self.model(out)
                loss += loss_fn(out, y2)
                count += 1

            else:
                x, y1, y2 = [x.half().cuda(non_blocking=True) for x in batch]
                with torch.cuda.amp.autocast():
                    out = self.model(x)
                    loss += loss_fn(out, y1)
                    out = self.model(out)
                    loss += loss_fn(out, y2)
                count += 1
        ema.restore() if ema is not None else None
        loss_val = loss.item() / count.item()
        return loss_val

    def pretrain_one_epoch(self, epoch, start_step, loss_fn, data_loader, optimizer, loss_scaler, ema=None):
        loss_val = torch.tensor(0., device=self.device)
        count = torch.tensor(1e-5, device=self.device)

        self.model.train()

        for step, batch in enumerate(data_loader):
            if step < start_step:
                continue
            if self.device == torch.device('cpu'):
                x, y = batch
                out = self.model(x) if self.args.model_name not in ['Transformer'] else self.model(x, y, epoch)
                loss = loss_fn(out, y)
                if torch.isnan(loss).int().sum() == 0:
                    count += 1
                    loss += loss
                loss.backward()
                optimizer.step()
            else:
                x, y = [x.half().cuda(non_blocking=True) for x in batch]

                with torch.cuda.amp.autocast():
                    out = self.model(x) if self.args.model_name not in ['Transformer'] else self.model(x, y, epoch)
                    loss = loss_fn(out, y)
                    if torch.isnan(loss).int().sum() == 0:
                        count += 1
                        loss_val += loss
                loss_scaler.scale(loss).backward()
                loss_scaler.step(optimizer)
                loss_scaler.update()
            ema.update() if ema is not None else None
            optimizer.zero_grad()

        return loss_val.item() / count.item()

    def finetune_one_epoch(self, epoch, start_step, loss_fn, data_loader, optimizer, loss_scaler, ema=None):
        loss_val = torch.tensor(0., device=self.device)
        count = torch.tensor(1e-5, device=self.device)

        self.model.train()

        for step, batch in enumerate(data_loader):
            if step < start_step:
                continue
            if self.device == torch.device('cpu'):
                x, y1, y2 = batch
                out = self.model(x) if self.args.model_name not in ['Transformer'] else self.model(x, y1, epoch)
                loss = loss_fn(out, y1)
                out = self.model(out)
                loss += loss_fn(out, y2)
                if torch.isnan(loss).int().sum() == 0:
                    count += 1
                    loss_val += loss
                loss.backward()
                optimizer.step()
            else:
                x, y1, y2 = [x.half().cuda(non_blocking=True) for x in batch]
                with torch.cuda.amp.autocast():
                    out = self.model(x) if self.args.model_name not in ['Transformer'] else self.model(x, y1, epoch)
                    loss = loss_fn(out, y1)
                    out = self.model(out) if self.args.model_name not in ['Transformer'] else self.model(out, y2, epoch)
                    loss += loss_fn(out, y2)
                    if torch.isnan(loss).int().sum() == 0:
                        count += 1
                        loss_val += loss
                loss_scaler.scale(loss).backward()
                loss_scaler.step(optimizer)
                loss_scaler.update()
            ema.update() if ema is not None else None
            optimizer.zero_grad()

        return loss_val.item() / count.item()

    def train(self, finetune=False, mode='indirect'):
        ema = EMA(self.model, 0.999) if self.args.ema else None
        ema.register() if ema is not None else None
        param_sum, buffer_sum, all_size = getModelSize(self.model)
        print(f"Number of Parameters: {param_sum}, Number of Buffers: {buffer_sum}, Size of Model: {all_size:.4f} MB")
        optimizer = torch.optim.AdamW(self.model.parameters(), self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.95))
        loss_scaler = torch.cuda.amp.GradScaler(enabled=True)
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min')
        loss_fn = torch.nn.MSELoss()

        # load data
        if mode == 'indirect':
            dataset_train, battery_id_train = self.dataset.get_pretrain_dataset('train')
            dataset_vali, battery_id_vali = self.dataset.get_pretrain_dataset('vali')
        else:
            dataset_train, battery_id_train = self.dataset.get_direct_predict_data('train')
            dataset_vali, battery_id_vali = self.dataset.get_direct_predict_data('vali')
        dataloader_train = DataLoader(dataset_train, self.args.batch_size,
                                      num_workers=0, shuffle=True,
                                      pin_memory=True, drop_last=False)  # , num_workers=8, pin_memory=True
        dataloader_vali = DataLoader(dataset_vali, self.args.batch_size,
                                    num_workers=0, shuffle=False,
                                    pin_memory=True, drop_last=False)  # , num_workers=8, pin_memory=True

        # load model
        start_epoch, start_step, min_loss, early_stop_count = \
            load_model(self.model, ema, optimizer, lr_scheduler, loss_scaler,
                       self.args.save_path_model + f'/{self.args.data_name}_{self.args.model_name}_backbone_best_{self.seq_len}_{self.pred_len}.pt')
        print(self.args.save_path_model + f'/{self.args.model_name}_backbone_best_{self.seq_len}_{self.pred_len}.pt')
        print(f"\nStart pretrain for {self.args.pretrain_epochs} epochs, now {start_epoch}/{self.args.pretrain_epochs}, min_loss:{min_loss}")

        last_loss = min_loss    # save the last loss to judge whether the loss is increasing continuously
        patience = self.args.patience

        for epoch in range(start_epoch, self.args.pretrain_epochs):  # tqdm(iter, leave=True, position=0, desc='Pretrain'):
            t0 = time.time()
            train_loss = self.pretrain_one_epoch(epoch, start_step, loss_fn, dataloader_train, optimizer, loss_scaler, ema)
            t1 = time.time()
            start_step = 0

            val_loss = self.vali(dataloader_vali, loss_fn, ema)
            lr_scheduler.step(val_loss)

            print(f"Epoch {epoch}/{self.args.pretrain_epochs} | Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}, "
                  f"Time: {t1 - t0:.2f}s, Early stop: {early_stop_count}/3")
            if val_loss < min_loss:
                min_loss = val_loss
                save_model(self.model, epoch + 1, 0, optimizer, lr_scheduler, loss_scaler, min_loss, early_stop_count, ema,
                           self.args.save_path_model + f'/{self.args.data_name}_{self.args.model_name}_backbone_best_{self.seq_len}_{self.pred_len}.pt')
            else:
                if last_loss <= val_loss:
                    early_stop_count += 1
                else:
                    early_stop_count = 0
                last_loss = val_loss
                if early_stop_count >= patience:
                    print(f'Early Stop at Epoch {epoch}')
                    break
        # 如果没有保存的信息，则在从新开始训练时，或者从预训练到微调时，需要考虑初始化模型、优化器、学习率策略器等

        if finetune:
            assert mode == 'indirect', 'Finetune: Only indirect prediction mode is supported!'
            # load model
            if os.path.exists(self.args.save_path_model + f'/{self.args.model_name}_finetune_best_{self.seq_len}_{self.pred_len}.pt'):
                start_epoch, start_step, min_loss, early_stop_count = \
                    load_model(self.model, ema, optimizer, lr_scheduler, loss_scaler,
                               self.args.save_path_model + f'/{self.args.model_name}_finetune_best_{self.seq_len}_{self.pred_len}.pt')
            else:
                start_epoch, start_step, min_loss, early_stop_count = \
                    load_model(self.model, ema, optimizer, lr_scheduler, loss_scaler,
                               self.args.save_path_model + f'/{self.args.model_name}_backbone_best_{self.seq_len}_{self.pred_len}.pt')
                start_epoch, start_step, min_loss, early_stop_count = 0, 0, np.inf, 0
            print(f"\nStart finetune for {self.args.finetune_epochs} epochs, now {start_epoch}/{self.args.finetune_epochs}, min_loss:{min_loss}")
            last_loss = min_loss

            # load data
            dataset_finetune_train, battery_id_train = self.dataset.get_finetune_dataset('train')
            dataloader_finetune_train = DataLoader(dataset_finetune_train, self.args.batch_size,
                                             num_workers=0, shuffle=True,
                                             pin_memory=True, drop_last=False)
            dataset_finetune_vali, battery_id_vali = self.dataset.get_finetune_dataset('vali')
            dataloader_finetune_vali = DataLoader(dataset_finetune_vali, self.args.batch_size,
                                             num_workers=0, shuffle=False,
                                             pin_memory=True, drop_last=False)

            for epoch in range(start_epoch, self.args.finetune_epochs):
                t0 = time.time()
                train_loss = self.finetune_one_epoch(epoch, start_step, loss_fn, dataloader_finetune_train, optimizer, loss_scaler, ema)
                t1 = time.time()
                start_step = 0

                val_loss = self.finetune_vali(dataloader_finetune_vali, loss_fn, ema)
                lr_scheduler.step(val_loss)

                print(
                    f"Epoch {epoch}/{self.args.finetune_epochs} | Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}, "
                    f"Time: {t1 - t0:.2f}s, Early stop: {early_stop_count}/3")
                if val_loss < min_loss:
                    min_loss = val_loss
                    save_model(self.model, epoch + 1, 0, optimizer, lr_scheduler, loss_scaler, min_loss, early_stop_count, ema,
                               self.args.save_path_model + f'/{self.args.data_name}_{self.args.model_name}_finetune_best_{self.seq_len}_{self.pred_len}.pt')
                else:
                    if last_loss <= val_loss:
                        early_stop_count += 1
                    else:
                        early_stop_count = 0
                    last_loss = val_loss
                    if early_stop_count >= patience:
                        print(f'Early Stop at Epoch {epoch}')
                        break

    @torch.no_grad()
    def test(self, mode='indirect'):
        loss_val = torch.tensor(0., device=self.device)
        count = torch.tensor(1e-5, device=self.device)

        # switch to evaluation mode
        # self.model.cpu()
        self.model.eval()
        self.model.float()

        # load data
        if mode == 'indirect':
            dataset_test, battery_names_test = self.dataset.get_test_dataset()
        else:
            dataset_test, battery_names_test = self.dataset.get_direct_predict_data('test')
        dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=0, pin_memory=True)
        print(f"Start test for {self.args.data_name} batteries")
        for i, batch in enumerate(dataloader_test):
            x, y = [x.float().cuda() for x in batch]
            if mode == 'indirect':
                output, step = [], math.ceil(y.shape[1]/self.pred_len)
                for j in range(step):
                    x = self.model(x)
                    output.append(x)  # (1, n, 1)
                output = torch.cat(output, dim=1)[:, :y.shape[1]]
            else:
                output = self.model(x)
            loss = torch.nn.MSELoss()(output, y)
            if torch.isnan(loss).int().sum() == 0:
                count += 1
                loss_val += loss
                if i % 50 == 0:
                    print(f'Battery: {battery_names_test} | Step: {i} | Test loss: {loss:.6f}')
        print(f'Test loss: {loss_val.item() / count.item():.6f}\n\n')

    @torch.no_grad()
    def visualize(self, battery_id=None, start_cycle=None):
        """ Visualize iterative prediction of battery aging trajectory """
        # TODO: Drawing motion pictures for visualization.
        # load data
        dataset_test, battery_id_test = self.dataset.get_test_dataset(visualize=True)
        battery_id = random.randint(0, len(battery_id_test) - 1) if battery_id is None else battery_id
        if start_cycle is None:
            data_num = len(dataset_test[battery_id])
            start_cycle = random.randint(int(data_num/4), int(data_num*3/4))
        print(f'Visualize battery aging trajectory prediction.\n',
              f'Battery: {battery_id_test[battery_id]}, Start from cycle {start_cycle}')

        data = dataset_test[battery_id].get_raw_data()
        input, output = data[start_cycle:start_cycle+self.seq_len].unsqueeze(0).float().cuda(), []
        step = math.ceil((data.shape[0]-(start_cycle+self.seq_len))/self.pred_len)

        # load model
        if os.path.exists(self.args.save_path_model + f'/{self.args.data_name}_{self.args.model_name}_finetune_best_{self.seq_len}_{self.pred_len}.pt'):
            model_path = self.args.save_path_model + f'/{self.args.data_name}_{self.args.model_name}_finetune_best_{self.seq_len}_{self.pred_len}.pt'
        elif os.path.exists(self.args.save_path_model + f'/{self.args.data_name}_{self.args.model_name}_backbone_best_{self.seq_len}_{self.pred_len}.pt'):
            model_path = self.args.save_path_model + f'/{self.args.data_name}_{self.args.model_name}_backbone_best_{self.seq_len}_{self.pred_len}.pt'
        else:
            raise FileNotFoundError(f'/{self.args.model_name} are not trained yet.')
        print(f'Load model from {model_path}')
        parmas = torch.load(model_path, map_location='cpu')
        parmas = parmas['ema']['shadow'] if self.args.ema else parmas['model']
        self.model.load_state_dict(parmas)
        self.model.float()

        for i in range(step):
            output_temp = self.model(input)
            input = output_temp
            output.append(output_temp.squeeze(0).cpu().numpy())
            print(i)
        output = np.concatenate(output, axis=0)  # (n,1)

        # plot
        plt.figure(figsize=(12, 6))
        plt.plot(data, label='Raw Data')
        plt.plot(np.arange(start_cycle, start_cycle+self.seq_len), data[start_cycle:start_cycle+self.seq_len], label='Raw Input')
        pred_x = np.arange(start_cycle+self.seq_len, data.shape[0])
        plt.plot(pred_x, output[:len(pred_x)], label='Prediction')
        plt.legend()
        plt.xlabel('Cycle')
        plt.ylabel('Capacity')
        plt.title(f'Battery {battery_id_test[battery_id]} Aging Trajectory Prediction (Start from cycle {start_cycle})')
        plt.savefig(self.args.save_path_data + f'/{self.args.data_name}_{self.args.model_name}_visualize_{self.seq_len}_{self.pred_len}.png')
        plt.show()

    @torch.no_grad()
    def visualize_direct_pred(self, battery_id=None):
        """ Visualize direct prediction of RUL """
        # load data
        dataset_test, battery_id_test = self.dataset.get_direct_predict_data(mode='test', visualize=True)
        battery_id = random.randint(0, len(battery_id_test) - 1) if battery_id is None else battery_id
        print(f'Visualize direct prediction of RUL.\n Battery: {battery_id_test[battery_id]}')

        data, RUL_list = dataset_test[battery_id].get_raw_data()
        RUL_list = RUL_list[self.args.seq_len-1:]
        RUL_pred = []
        for i in range(len(data) - self.args.seq_len + 1):
            data_temp = data[i:i+self.args.seq_len].unsqueeze(0).float().cuda()
            RUL_pred.append(self.model(data_temp).squeeze(0).cpu().numpy())
        RUL_pred = np.concatenate(RUL_pred, axis=0)  # (n,1)

        # plot
        plt.figure(figsize=(12, 6))
        max_RUL = 2 if self.args.normalize else max(max(RUL_list), max(RUL_pred)) + 10
        if self.args.normalize:
            plt.plot(range(-3, max_RUL), range(-3, max_RUL), c='#B5B04C', linewidth=2)
        else:
            plt.plot(range(max_RUL), range(max_RUL), c='#B5B04C', linewidth=2)
        plt.scatter(RUL_list, RUL_pred, s=10)
        plt.xlabel('True RUL')
        plt.ylabel('Predicted RUL')
        plt.title(f'Battery {battery_id_test[battery_id]} RUL Prediction')
        plt.savefig(self.args.save_path_data + f'/{self.args.data_name}_{self.args.model_name}_visualize_direct_pred_{self.seq_len}_{self.pred_len}.png')
        plt.show()




if __name__ == '__main__':

    class AttrDict(dict):
        def __getattr__(self, name):
            if name in self:
                return self[name]
            else:
                return None
            # raise AttributeError(f"'AttrDict' object has no attribute '{name}'")

        def __setattr__(self, name, value):
            self[name] = value

    args = AttrDict(dict({'model_name': 'DLinear', 'individual': False,
                          'use_gpu': True, 'gpu': 0, 'pretrain_epochs': 1000, 'finetune_epochs': 150, 'batch_size': 32,
                          'data_name': 'RWTH', 'seq_len': 12, 'pred_len': 1, 'in_dim': 1, 'out_dim': 1, 'normalize': True,
                          'lr': 0.0001, 'weight_decay': 0.05, 'patience': 5, 'ema': True, 'seed': 42,
                          'save_path_model': 'checkpoints/direct', 'save_path_data': 'data_result',
                          })
                    )

    # args = AttrDict(dict({'model_name': 'GRU', 'n_layer': 2, 'hidden_dim': 20, 'dropout': 0.1, 'bidirect': False,
    #                       'use_gpu': True, 'gpu': 0, 'pretrain_epochs': 200, 'finetune_epochs': 150, 'batch_size': 32,
    #                       'data_name': 'RWTH', 'seq_len': 10, 'pred_len': 1, 'in_dim': 1, 'out_dim': 1,
    #                       'normalize': True,
    #                       'lr': 0.0001, 'weight_decay': 0.05, 'patience': 5, 'ema': True, 'seed': 42,
    #                       'save_path_model': 'checkpoints/directs', 'save_path_data': 'data_result',
    #                       })
    #                 )

    # args = AttrDict(dict({'model_name': 'Transformer', 'batch_size': 32, 'pretrain_epochs': 200, 'finetune_epochs': 150,
    #                       'use_gpu': True, 'gpu': 0, 'data_name': 'RWTH',
    #                       'seq_len': 10, 'pred_len': 1, 'in_dim': 1, 'out_dim': 1, 'normalize': True,
    #                       'nhead': 2, 'd_model': 4, 'n_encoder_layer': 2, 'n_decoder_layer': 2, 'feed_hidden_dim': 30,
    #                       'lr': 0.0001, 'weight_decay': 0.05, 'patience': 5, 'ema': False, 'seed': 42,
    #                       'save_path_model': 'checkpoints/direct', 'save_path_data': 'data_result',
    #                       })
    #                 )

    exp = ExpMain(args)

    # print(exp.model.get_model_info())


    # exp.train(finetune=True)
    # exp.test()
    # exp.visualize()

    exp.train(finetune=False, mode='direct')
    # exp.test(mode='direct')
    exp.visualize_direct_pred(battery_id=3)
