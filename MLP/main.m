clc 
clear  all
% ===== Inputs ===== %
load Wenchang.mat

train_set_inputs = train_x;
train_labels = round(train_y*10000);
train_set_goals = labels2goals(train_labels, 4584);
test_set_inputs = test_x;
test_labels = round(test_y*10000);
test_set_goals = labels2goals(test_labels,4584);

hidden_layers_sizes = [1200 800];
activation_function = 'tanh';
number_of_epochs = 30;
learning_rate = 0.01;
batch_size = 2;


% ===== Initializations ===== %

[~, number_of_hidden_layers] = size(hidden_layers_sizes);
[number_of_examples, input_layer_size] = size(train_set_inputs);
[~, output_layer_size] = size(train_set_goals);
[number_of_tests, ~] = size(test_set_inputs);

weights_and_biases = cell(number_of_hidden_layers+1, 1);
desired_weights_and_biases = cell(number_of_hidden_layers+1, 1);
for l=1:number_of_hidden_layers+1
    desired_weights_and_biases(l) = {0};
end

% randomize weights with gaussian (mean = 0, standard deviation = 1)
% random_number_of_layer = random_number / number_of_neurons_in_layer

rng(345); % set seed for reproducibility

rand_multiplier = 1 / input_layer_size;
weights_and_biases(1) = {rand_multiplier * normrnd(0, 1, [hidden_layers_sizes(1), input_layer_size+1])}; % weights and biases between input layer and first hidden layer, +1 for the bias, matrix: next_layer_size x current_layer_size
for i=2:number_of_hidden_layers
    rand_multiplier = 1 / hidden_layers_sizes(i-1);
    weights_and_biases(i) = {rand_multiplier * normrnd(0, 1, [hidden_layers_sizes(i), hidden_layers_sizes(i-1)+1])}; % weights and biases between hidden layers, +1 for the bias, matrix: next_layer_size x current_layer_size
end
rand_multiplier = 1 / hidden_layers_sizes(end);
weights_and_biases(end) = {rand_multiplier * normrnd(0, 1, [output_layer_size, hidden_layers_sizes(end)+1])}; % weights and biases between last hidden layer and output layer, +1 for the bias, matrix: next_layer_size x current_layer_size



% ===== Training ===== %


weighted_outputs = cell(number_of_hidden_layers + 1, 1); % z = w*a + ... + b
squishified_weighted_outputs = cell(number_of_hidden_layers + 1, 1); % a = phi(z)

training_errors = ones(number_of_epochs, 1);
testing_errors = ones(number_of_epochs, 1);
training_precisions = ones(number_of_epochs, 1);
testing_precisions = ones(number_of_epochs, 1);


fprintf(1,'Training...\n');
start_time = cputime;

for epoch=1:number_of_epochs
    
    sum_train_errors = 0;
    sum_test_errors = 0;
    
    % Shuffle train_set_inputs with train_set_goals %
    
    shuffler = randperm(number_of_examples);
    train_set_inputs = train_set_inputs(shuffler, :);
    train_set_goals = train_set_goals(shuffler, :);
    
    for p=1:number_of_examples

        current_example_with_bias = [train_set_inputs(p, :) 1]';
        current_goals = train_set_goals(p, :)';

        % Feed Forward (Calculation of neuron outputs) %

        z = cell2mat(weights_and_biases(1)) * current_example_with_bias;

        weighted_outputs(1) = {z};
        squishified_weighted_outputs(1) = {phi(z, activation_function)}; % outputs of neurons from the first hidden layer

        for l=2:number_of_hidden_layers+1
            z = cell2mat(weights_and_biases(l)) * [cell2mat(squishified_weighted_outputs(l-1))' 1]';
            weighted_outputs(l) = {z};
            squishified_weighted_outputs(l) = {phi(z, activation_function)};
        end   

        error = current_goals - cell2mat(squishified_weighted_outputs(end));
        sum_train_errors = sum_train_errors + sumsqr(error);
        
        % Back propagation (Calculation of desired weights) %

        sigma = phi_d(cell2mat(weighted_outputs(end)), activation_function);
        alpha = cell2mat(squishified_weighted_outputs(end-1));

        delta = sigma .* error;
        
        delta_scaled = delta(:, ones(hidden_layers_sizes(end) + 1, 1)); % scaling for every neuron of the last hidden layer
        alpha_scaled = [alpha(:, ones(output_layer_size, 1))' ones(output_layer_size, 1)]; % scaling for every neuron of the output layer
        % accumulate weights of current batch
        desired_weights_and_biases(end) = {cell2mat(desired_weights_and_biases(end)) + delta_scaled .* alpha_scaled};

        for l=number_of_hidden_layers:-1:1

            sigma = phi_d(cell2mat(weighted_outputs(l)), activation_function); 
            if l>1
                alpha = cell2mat(squishified_weighted_outputs(l-1));
            else % previous layer is input layer
                alpha = current_example_with_bias;
            end
            next_layer_weights_and_biases = cell2mat(weights_and_biases(l+1));
            next_layer_weights = next_layer_weights_and_biases(:, 1:hidden_layers_sizes(l)); % crop the last column (biases)
            previous_delta_rescaled = delta(:, ones(1, hidden_layers_sizes(l)));
            next_layer_error = sum(next_layer_weights' .* previous_delta_rescaled', 2);
                        

            delta = sigma .* next_layer_error;
            if l>1
                delta_scaled = delta(:, ones(hidden_layers_sizes(l-1) + 1, 1)); % scaling for every neuron of the current hidden layer, +1 for bias
                alpha_scaled = [alpha(:, ones(hidden_layers_sizes(l), 1))' ones(hidden_layers_sizes(l), 1)]; % scaling for every neuron of the next hidden layer
            else % previous layer is input layer
                delta_scaled = delta(:, ones(input_layer_size + 1, 1)); % scaling for every neuron of the input layer
                alpha_scaled = alpha(:, ones(hidden_layers_sizes(l), 1))'; % scaling for every neuron of the next hidden layer
            end
            % accumulate weights of current batch
            desired_weights_and_biases(l) = {cell2mat(desired_weights_and_biases(l)) + delta_scaled .* alpha_scaled};

        end

        % Update weights if we have reached the end of a batch or the end of inputs%

        if (mod(p-1, batch_size) == batch_size -1 || p==number_of_examples)
            for k=1:number_of_hidden_layers+1
                weights_and_biases(k) = {cell2mat(weights_and_biases(k)) + learning_rate .* cell2mat(desired_weights_and_biases(k))/batch_size};
                desired_weights_and_biases(k) = {0};
            end
        end


    end
    
    training_errors(epoch) = sum_train_errors / number_of_examples;
    




    % ===== Testing ===== %

    
   
    predict=zeros(number_of_tests,1);
    for p=1:number_of_tests

        current_test_with_bias = [test_set_inputs(p, :) 1]';
        current_goals = test_set_goals(p, :)';

        % Feed Forward (Calculation of neuron outputs) %

        current_weighted_outputs = phi(cell2mat(weights_and_biases(1)) * current_test_with_bias, activation_function); % outputs of neurons from the first hidden layer

        for l=2:number_of_hidden_layers+1
            current_weighted_outputs = phi(cell2mat(weights_and_biases(l)) * [current_weighted_outputs; 1], activation_function);
        end

        
        [~, max_neuron_id] = max(current_weighted_outputs);
        predict(p,1) = (max_neuron_id -1)/10000;
        end
      

end
   
    figure 
    plot (predict);
    hold on
    plot (test_y);
    hold on



training_time = cputime - start_time;
fprintf(1,'Seconds: %g\n', training_time); 














       