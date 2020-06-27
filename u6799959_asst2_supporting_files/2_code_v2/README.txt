There are several requirement/tips for running the code:
(1) Operating system: Win10
(2) File Folder 'face-emotion_v2'  under the current working directory of you IDE. 
(3) Training Mode: Pytorch GPU Training 
(4) To avoid keyboard Interrupt  while running, please go to your File->Settings->Building, Execution, Deployment->Python Debugger of your IDE,
and turn off the 'Attach to subprocess automatically while debugging'.
(5) There are mainly 6 parts of the code,  line 25-82 is for creating training and testing datasets, line 83-162 is for data transformation & data loading 
of training/testing datasets, and visualization one batch of samples in training dataloader. Line 165-299 is for impelmenting training-testing for certain
set of hyperparameters, line 303-415 is for visualizing the performance of the trained model. Line 417-597 is for the cross-validation of certain set of 
hyperparameters and finally line 598-636 is for model performance evaluation. 
(7) Thanks for reading my paper&code, if you have any question, please feel free to contact u6799959@anu.edu.au
