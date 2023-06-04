"""Trainer.

@author: Zhenye Na - https://github.com/Zhenye-Na
@reference: "End to End Learning for Self-Driving Cars", arXiv:1604.07316
"""

import os
import torch
import numpy as np
import pandas as pd
from utils import toDevice
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import cv2
import torchvision.transforms as transforms



class Trainer(object):
    """Trainer class."""

    def __init__(self,
                 ckptroot,
                 model,
                 device,
                 epochs,
                 criterion,
                 optimizer,
                 scheduler,
                 start_epoch,
                 trainloader,
                 validationloader,
                 test,
                 csv_name,
                 plot):
        """Self-Driving car Trainer.

        Args:
            model: CNN model
            device: cuda or cpu
            epochs: epochs to training neural network
            criterion: nn.MSELoss()
            optimizer: optim.Adam()
            start_epoch: 0 or checkpoint['epoch']
            trainloader: training set loader
            validationloader: validation set loader

        """
        super(Trainer, self).__init__()

        self.model = model
        self.device = torch.device('cuda')
        self.epochs = epochs
        self.ckptroot = ckptroot
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.start_epoch = start_epoch
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.test = test
        self.csv_name = csv_name
        self.plot = plot

    def train(self):
        """Training process."""
        import torch.nn as nn
        #self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        train_loss_list = [] #to plot the train_loss graph
        validation_loss_list = [] #to plot the validation_loss graph

        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            self.scheduler.step()
            if self.test == False: #if learning
                print("<Training Epoch {} ...>".format(epoch))
                # Training
                train_loss = 0.0
                self.model.train()
                for local_batch, (centers) in enumerate(self.trainloader):
                    # Transfer to GPU
                    centers = toDevice(centers, self.device)
                    # Model computations
                    self.optimizer.zero_grad()
                    datas = [centers]
                    #in a batch
                    for data in datas:
                        imgs, angles = data
                        outputs = self.model(imgs)
                        
                        loss = self.criterion(outputs, angles.unsqueeze(1))
                        loss.backward()
                        self.optimizer.step()
                        train_loss += loss.data.item()
                        #print("loss: ",loss.data.item())
                
                    # if local_batch % 100 == 0:
                    #     print("Training Epoch: {} | Loss: {}".format(
                    #         epoch, train_loss / (local_batch + 1)))

                    #     train_loss_list.append(train_loss / (local_batch + 1))
                print("   Train Loss: {}".format(train_loss / (local_batch + 1)))
                train_loss_list.append(train_loss / (local_batch + 1))
            
            #validation
            self.model.eval()
            valid_loss = 0
            test_loss = 0

            output_angle_list = [] #모델이 예측한 출력값을 저장하기 위한 리스트
            with torch.set_grad_enabled(False):
                for local_batch, (centers) in enumerate(self.validationloader):
                    # Transfer to GPU
                    centers = toDevice(centers, self.device)
                    
                    # Model computations
                    self.optimizer.zero_grad()
                    datas = [centers]

                    for data in datas:
                        imgs, angles = data
                        outputs = self.model(imgs)
                        output = outputs.cpu().numpy()  #to get output angle form of list
                        loss = self.criterion(outputs, angles.unsqueeze(1))

                        if(self.test == True): #if test case, print output angle by frame and save
                            for output_angle in output:
                                
                                # output_angle[0] = output_angle[0]-0.253
                                # output_angle[0] = output_angle[0]-0.01
                                

                                # if output_angle[0] < -0.875 :
                                #      output_angle[0] = -1.00
                                # elif output_angle[0] < -0.625 :
                                #      output_angle[0] = -0.75
                                # elif output_angle[0] < -0.375 :
                                #      output_angle[0] = -0.50
                                # elif output_angle[0] < -0.125 :
                                #      output_angle[0] = -0.25
                                # elif output_angle[0] < 0.125 :
                                #      output_angle[0] = 0.00
                                # elif output_angle[0] < 0.375 :
                                #      output_angle[0] = 0.25
                                # elif output_angle[0] < 0.625 :
                                #      output_angle[0] = 0.5
                                # elif output_angle[0] < 0.875 :
                                #      output_angle[0] = 0.75
                                # else : 
                                #      output_angle[0] = 1.00
                                
                                output_angle_list.append(output_angle[0])
                                # 상곤
                                #print(output_angle[0])
                            
                        valid_loss += loss.data.item()
                   
                    # if local_batch % 100 == 0:
                    #     if(self.test == False): #학습하는 경우에는 Validation Loss를 출력하도록
                    #         print("Validation Loss : {}".format(valid_loss / (local_batch + 1)))
                    #         validation_loss_list.append (valid_loss / (local_batch + 1))
                    #     else:	#테스트 하는경우에는 test_loss를 저장하도록
                    #         test_loss = valid_loss / (local_batch+1)

                if(self.test == True):
                    test_loss = valid_loss / (local_batch+1)
                else:
                    print("   Validation Loss : {}".format(valid_loss / (local_batch + 1)))
                    validation_loss_list.append (valid_loss / (local_batch + 1))                    
            
            #test하는 경우 output값을 csv파일로 저장
            if(self.test == True):
 
                print("==> test done")
                print("Test Loss: ", test_loss)
				
				#driving_log_list = pd.read_csv()
                self.plot_output(output_angle_list)			#결과를 Graph로 Plot함
                dataframe = pd.DataFrame(output_angle_list)	
                dataframe.to_csv("./output_angle.csv", header=False, index=False)	#결과를 output_angle.csv파일로 저장
                #print("----------local batch end-------------")                        
            
            print()
            # Save model
            if self.test == False:
                if epoch % 50 == 0 or epoch == self.epochs + self.start_epoch - 1:

                    print("==> Save checkpoint ...")

                    state = {
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                    }

                    self.save_checkpoint(state)
        
        #학습이 끝나면 Loss Graph를 그리기
        if self.plot == True and self.test==False:
            train_loss_list = np.array(train_loss_list)
            validation_loss_list = np.array(validation_loss_list)
            self.plot_train_loss(train_loss_list, validation_loss_list)
		
    def save_checkpoint(self, state):
        """Save checkpoint."""
        if not os.path.exists(self.ckptroot):
            os.makedirs(self.ckptroot)

        torch.save(state, self.ckptroot + 'model-{}.h5'.format(state['epoch']))
	
	#train loss그래프를 그려주는 함수
    def plot_train_loss(self, train_loss_list, validation_loss_list):
        """"Plot Loss Graph."""
        plt.title("Training Loss vs validation Loss")
        plt.xlabel("Epcoh")
        plt.plot(range(len(train_loss_list)), train_loss_list, 'b', label='Training Loss')
        plt.plot(range(len(train_loss_list)), validation_loss_list, 'g', label='Validation Loss')
        #plt.plot(np.linspace(0, len(train_loss_list), len(validation_loss_list)), validation_loss_list, 'g-.', label='Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
	
	#output 그래프를 그려주는 함수
    def plot_output(self, output_angle_list):
        gt_csv = self.csv_name
        gt_csv = ""+gt_csv
        df_gt = pd.read_csv(gt_csv, names=['center', 'steering'])
        gt = df_gt['steering']

        #gt_csv = gt_csv.split('/')[-1]
        plt.title("Ground Truth vs Output Angle")
        plt.ylabel("output angles")
        plt.xlabel("frame no")
        plt.scatter(range(len(gt)), gt,color='b' ,s=5, label = "Ground Truth")
        plt.scatter(range(len(output_angle_list)), output_angle_list, color='g', s=5, label = "Model Output")
        plt.legend()

		#plt.scatter(range(len(output_angle_list)), driving_og_list, c=2)
        plt.show()
