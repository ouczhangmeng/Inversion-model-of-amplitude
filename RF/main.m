clc
clear all
%% Load
load Wenchang.mat

P_train = train_x;
P_test = test_x;
T_train = train_y;
T_test = test_y;

%% Train
model = classRF_train(P_train,T_train,500,7);

%% Test
[T_sim,votes] = classRF_predict(P_test,model);
%% Result
figure
plot (T_sim);
hold on
plot (T_test);
hold on
    


