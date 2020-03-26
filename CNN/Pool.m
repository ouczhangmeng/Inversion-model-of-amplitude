function y = Pool(x)
%     
% 1*1 mean pooling    
%
%
[xrow, xcol, numFilters] = size(x);

y = zeros(xrow/1, xcol/1, numFilters);    
for k = 1:numFilters
  filter = ones(1) / (1*1);    % for mean    
  image  = conv2(x(:, :, k), filter, 'valid');
  
  y(:, :, k) = image(1:1:end, 1:1:end);
end

end
 