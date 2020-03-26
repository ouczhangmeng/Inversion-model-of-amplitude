clear all
load Wenchang.mat
Images = train_x';
Images = reshape(Images, 1, 5, []);
Labels =  round(train_y*10000);
Labels(Labels == 0) = 4584;    % 0 --> 10

rng(3);

% Learning
%
W1 = 1e-2*randn([1 1 20]);
W5 = (2*rand(100, 100) - 1) * sqrt(6) / sqrt(360 + 100);
Wo = (2*rand( 4584,  100) - 1) * sqrt(6) / sqrt( 4584 +  100);

X = Images;
D = Labels;

for epoch = 1:4
  epoch
  [W1, W5, Wo] = MnistConv(W1, W5, Wo, X, D);
end

save('MnistConv.mat');


% Test
%
X = test_x;
X = reshape(X, 1, 5, []);
D = round(test_y*10000);

acc = 0;
N   = length(D);
predict=zeros(N,1);
for k = 1:N
  x = X(:, :, k);                   % Input,              1*5

  y1 = Conv(x, W1);                 % Convolution,      1*5*20
  y2 = ReLU(y1);                    %
  y3 = Pool(y2);                    % Pool,             1*5*20
  y4 = reshape(y3, [], 1);          %                    100*1
  v5 = W5*y4;                       % ReLU,                    
  y5 = ReLU(v5);                    %
  v  = Wo*y5;                       % Softmax,           4584
  y  = Softmax(v);                  %

  [~, i] = max(y);
predict(k,1) = i;
end
predict=predict/10000;
test=D/10000;
figure 
    plot (predict);
    hold on
    plot (test);
    hold on
legend('predict','test');

