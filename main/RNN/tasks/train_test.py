import os, time, copy
import torch
import torch.nn as nn

class Train_Test():
    def __init__(self,  train_loader, valid_loader, test_loader, input_size, device='cuda'): 
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.input_size = input_size
        self.device = device

    def train(self, model, dataloaders, criterion, num_epochs, optimizer):
        since = time.time() 
        
        train_loss_history = []
        val_loss_history = []

        best_model_wts = copy.deepcopy(model.state_dict()) # 모델의 초기 Weight값 (각 Layer 별 초기 Weight값이 저장되어 있음)
        best_loss = 999999999 # MSE는 작을수록 좋은 metric이므로, 초기 높은 값에서 갱신
        best_epoch = 0

        for epoch in range(num_epochs):
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print('\nEpoch {}/{}'.format(epoch + 1, num_epochs))

            # 각 epoch마다 순서대로 training과 validation을 진행
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 모델을 training mode로 설정
                else:
                    model.eval()   # 모델을 validation mode로 설정

                running_loss = 0.0
                running_total = 0

                # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
                for data in dataloaders[phase]:
                    inputs = data['X'].to(self.device)
                    targets = data['Y'].to(self.device)
                    # seq_lens = seq_lens.to(self.parameter['device'])
                    
                    # parameter gradients를 0으로 설정
                    optimizer.zero_grad()

                    # forward
                    # training 단계에서만 gradient 업데이트 수행
                    with torch.set_grad_enabled(phase == 'train'):
                        # input을 model에 넣어 output을 도출한 후, loss를 계산함
                        outputs = model(inputs)
                        
                        loss = criterion(outputs.to(torch.float32), targets.to(torch.float32))

                        # backward (optimize): training 단계에서만 수행
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # batch별 loss를 축적함
                    running_loss += loss.item() * inputs.size(0)
                    running_total += targets.size(0)

                # epoch의 loss 및 accuracy 도출
                epoch_loss = running_loss / running_total
                                    
                if epoch == 0 or (epoch + 1) % 10 == 0:
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # validation 단계에서 validation loss가 감소할 때마다 best model 가중치를 업데이트함
                if phase == 'val' and epoch_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                elif phase == 'val':
                    val_loss_history.append(epoch_loss)

        # 전체 학습 시간 계산 (학습이 완료된 후)
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val MSE: {:4f}'.format(best_loss))

        # validation loss가 가장 낮았을 때의 best model 가중치를 불러와 best model을 구축함
        model.load_state_dict(best_model_wts)

        return model, train_loss_history, val_loss_history, best_epoch

    def test(self, model, dataloaders):
        model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            for phase in ['train', 'val', 'test']:

                preds = []
                y_true = []
                
                for data in dataloaders[phase]:
                    inputs = data['X'].to(self.device)
                    targets = data['Y'].to(self.device)

                    # forward
                    # input을 model에 넣어 output을 도출
                    pred = model(inputs)
                    
                    preds.extend(pred.detach().cpu().numpy())
                    y_true.extend(targets.detach().cpu().numpy())

                if phase == "train":
                    train_preds = torch.tensor(preds)
                    train_y_true = torch.tensor(y_true)
                if phase == "val":
                    val_preds = torch.tensor(preds)
                    val_y_true = torch.tensor(y_true)
                if phase == "test":
                    test_preds = torch.tensor(preds)
                    test_y_true = torch.tensor(y_true)
            
            # mse = nn.MSELoss()(preds, y_true).item()

        return train_preds, train_y_true, val_preds, val_y_true, test_preds, test_y_true 
    
class Train_Test_Attention():
    def __init__(self,  train_loader, valid_loader, test_loader, input_size, num_classes, device='cuda'): ##### config는 jupyter 파일을 참고
        """
        Initialize Train_Test class

        :param config: configuration
        :type config: dictionary

        :param train_loader: train dataloader
        :type config: DataLoader

        :param valid_loader: validation dataloader
        :type config: DataLoader

        :param test_loader: test dataloader
        :type config: DataLoader
        """

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.input_size = input_size
        self.num_classes = num_classes

        self.device = device

    def train(self, model, dataloaders, criterion, num_epochs, optimizer):
        """
        Train the model

        :param model: initialized model
        :type model: model

        :param dataloaders: train & validation dataloaders
        :type dataloaders: dictionary

        :param criterion: loss function for training
        :type criterion: criterion

        :param num_epochs: the number of train epochs
        :type num_epochs: int

        :param optimizer: optimizer used in training
        :type optimizer: optimizer

        :return: trained model
        :rtype: model
        """

        since = time.time() 
        
        train_loss_history = []
        val_loss_history = []

        best_model_wts = copy.deepcopy(model.state_dict()) # 모델의 초기 Weight값 (각 Layer 별 초기 Weight값이 저장되어 있음)
        best_loss = 999999999

        for epoch in range(num_epochs):
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print()
                print('Epoch {}/{}'.format(epoch + 1, num_epochs))

            # 각 epoch마다 순서대로 training과 validation을 진행
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 모델을 training mode로 설정
                else:
                    model.eval()   # 모델을 validation mode로 설정

                running_loss = 0.0
                running_total = 0

                # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
                for data in dataloaders[phase]:
                    inputs = data['X'].to(self.device)
                    targets = data['Y'].to(self.device)

                    # seq_lens = seq_lens.to(self.parameter['device'])
                    
                    # parameter gradients를 0으로 설정
                    optimizer.zero_grad()

                    # forward
                    # training 단계에서만 gradient 업데이트 수행
                    with torch.set_grad_enabled(phase == 'train'):
                        # input을 model에 넣어 output을 도출한 후, loss를 계산함
                        outputs, attn_scores = model(inputs)
                        
                        loss = criterion(outputs.to(torch.float32), targets.to(torch.float32))

                        # backward (optimize): training 단계에서만 수행
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # batch별 loss를 축적함
                    running_loss += loss.item() * inputs.size(0)
                    running_total += targets.size(0)

                # epoch의 loss 및 accuracy 도출
                epoch_loss = running_loss / running_total
                                    
                if epoch == 0 or (epoch + 1) % 10 == 0:
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # validation 단계에서 validation loss가 감소할 때마다 best model 가중치를 업데이트함
                if phase == 'val' and epoch_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                elif phase == 'val':
                    val_loss_history.append(epoch_loss)

        # 전체 학습 시간 계산 (학습이 완료된 후)
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val MSE: {:4f}'.format(best_loss))

        # validation loss가 가장 낮았을 때의 best model 가중치를 불러와 best model을 구축함
        model.load_state_dict(best_model_wts)

        return model, train_loss_history, val_loss_history, best_epoch  # , attn_scores

    def test(self, model, dataloaders):

        model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            for phase in ['train', 'val', 'test']:
                preds = []
                y_true = []
                attn_scores = []

                for data in dataloaders[phase]:
                    inputs = data['X'].to(self.device)
                    targets = data['Y'].to(self.device)

                    # forward
                    # input을 model에 넣어 output을 도출
                    pred, attn_score = model(inputs)
                    
                    preds.extend(pred.detach().cpu().numpy())
                    y_true.extend(targets.detach().cpu().numpy())
                    attn_scores.extend(attn_score.detach().cpu().tolist())
                
                if phase == "train":
                    train_preds = torch.tensor(preds)
                    train_y_true = torch.tensor(y_true)
                    train_attn_scores = torch.tensor(attn_scores)
                if phase == "val":
                    val_preds = torch.tensor(preds)
                    val_y_true = torch.tensor(y_true)
                    val_attn_scores = torch.tensor(attn_scores)
                if phase == "test":
                    test_preds = torch.tensor(preds)
                    test_y_true = torch.tensor(y_true)
                    test_attn_scores = torch.tensor(attn_scores)
            
            # mse = nn.MSELoss()(preds, y_true).item()
            
        return train_preds, train_y_true, train_attn_scores, val_preds, val_y_true, val_attn_scores, test_preds, test_y_true, test_attn_scores