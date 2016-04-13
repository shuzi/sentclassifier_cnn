model = nn.Sequential()
if fbok then 
   model:add(nn.LookupTableGPU(mapWordIdx2Vector:size()[1], opt.embeddingDim))
else
   model:add(nn.LookupTable(mapWordIdx2Vector:size()[1], opt.embeddingDim))
end
model:add(nn.View(opt.batchSize*trainDataTensor:size()[2], opt.embeddingDim))
model:add(nn.Linear(opt.embeddingDim, opt.wordHiddenDim))
model:add(nn.View(opt.batchSize, trainDataTensor:size()[2], opt.wordHiddenDim))
model:add(nn.Tanh())
if cudnnok then
   model:add(cudnn.TemporalConvolution(opt.wordHiddenDim, opt.numFilters, opt.contConvWidth))
elseif fbok then
   model:add(nn.TemporalConvolutionFB(opt.wordHiddenDim, opt.numFilters, opt.contConvWidth))
else
   model:add(nn.TemporalConvolution(opt.wordHiddenDim, opt.numFilters, opt.contConvWidth))
end
model:add(nn.Max(2))
model:add(nn.Tanh())
model:add(nn.Linear(opt.numFilters, opt.hiddenDim))
model:add(nn.Tanh())
model:add(nn.Linear(opt.hiddenDim, opt.numLabels))
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
--cudnn.convert(model, cudnn)
model:get(1).weight:copy(mapWordIdx2Vector)

model_test = nn.Sequential()
if fbok then
   model_test:add(nn.LookupTableGPU(mapWordIdx2Vector:size()[1], opt.embeddingDim))
else
   model_test:add(nn.LookupTable(mapWordIdx2Vector:size()[1], opt.embeddingDim))
end
model_test:add(nn.View(opt.batchSizeTest*validDataTensor:size()[2], opt.embeddingDim))
model_test:add(nn.Linear(opt.embeddingDim, opt.wordHiddenDim))
model_test:add(nn.View(opt.batchSizeTest, validDataTensor:size()[2], opt.wordHiddenDim))
model_test:add(nn.Tanh())
if cudnnok then
   model_test:add(cudnn.TemporalConvolution(opt.wordHiddenDim, opt.numFilters, opt.contConvWidth))
elseif fbok then
   model_test:add(nn.TemporalConvolutionFB(opt.wordHiddenDim, opt.numFilters, opt.contConvWidth))
else
   model_test:add(nn.TemporalConvolution(opt.wordHiddenDim, opt.numFilters, opt.contConvWidth))
end
model_test:add(nn.Max(2))
model_test:add(nn.Tanh())
model_test:add(nn.Linear(opt.numFilters, opt.hiddenDim))
model_test:add(nn.Tanh())
model_test:add(nn.Linear(opt.hiddenDim, opt.numLabels))
model_test:add(nn.LogSoftMax())

model_test:get(1).weight = model:get(1).weight
model_test:get(3).weight = model:get(3).weight
model_test:get(3).bias = model:get(3).bias
model_test:get(6).weight = model:get(6).weight
model_test:get(6).bias = model:get(6).bias
model_test:get(9).weight = model:get(9).weight
model_test:get(9).bias = model:get(9).bias
model_test:get(11).weight = model:get(11).weight
model_test:get(11).bias = model:get(11).bias


if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
   model_test:cuda()
   model_test:get(1).weight = model:get(1).weight
   model_test:get(3).weight = model:get(3).weight
   model_test:get(3).bias = model:get(3).bias
   model_test:get(6).weight = model:get(6).weight
   model_test:get(6).bias = model:get(6).bias
   model_test:get(9).weight = model:get(9).weight
   model_test:get(9).bias = model:get(9).bias
   model_test:get(11).weight = model:get(11).weight
   model_test:get(11).bias = model:get(11).bias
end
if model then
   parameters,gradParameters = model:getParameters()
   print("Model Size: ", parameters:size()[1])
   parametersClone = parameters:clone()
end
print(model)
print(criterion)

if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      learningRateDecay = opt.learningRateDecay,
      momentum = opt.momentum,
      learningRateDecay = 0,
      dampening = 0,
      nesterov = opt.nesterov
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = trsize * opt.t0
   }
   optimMethod = optim.asgd

else
   error('unknown optimization method')
end

function saveModel(s)
   torch.save(opt.outputprefix .. string.format("_%010.2f_model", s), parameters)
end

function loadModel(m)
   parameters:copy(torch.load(m))
end

function cleanMemForRuntime()
   parametersClone = nil
   gradParameters = nil
   model_test:get(1).gradWeight = nil 
   model:get(1).gradWeight = nil
   model_test:get(3).gradWeight = nil 
   model:get(3).gradWeight = nil 
   model_test:get(3).gradBias = nil 
   model:get(3).gradBias = nil
   model_test:get(6).gradWeight = nil 
   model:get(6).gradWeight = nil
   model_test:get(6).gradBias = nil
   model:get(6).gradBias = nil
   model_test:get(9).gradWeight = nil 
   model:get(9).gradWeight = nil
   model_test:get(9).gradBias = nil
   model:get(9).gradBias = nil
   model_test:get(11).gradWeight = nil 
   model:get(11).gradWeight = nil
   model_test:get(11).gradBias = nil
   model:get(11).gradBias = nil
   collectgarbage()
   collectgarbage()
end


function train()
    epoch = epoch or 1
    if optimState.evalCounter then
        optimState.evalCounter = optimState.evalCounter + 1
    end
    local time = sys.clock()
    model:training()
    local batches = trainDataTensor:size()[1]/opt.batchSize
    local bs = opt.batchSize
    shuffle = torch.randperm(batches)
    for t = 1,batches,1 do
        local begin = (shuffle[t] - 1)*bs + 1
        local input = trainDataTensor:narrow(1, begin , bs) 
        local target = trainDataTensor_y:narrow(1, begin , bs)
        
        local feval = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end
            gradParameters:zero()
            local f = 0
            local output = model:forward(input)
            f = criterion:forward(output, target)
            local df_do = criterion:backward(output, target)
            model:backward(input, df_do) 
            --cutorch.synchronize()
            if opt.L1reg ~= 0 then
               local norm, sign = torch.norm, torch.sign
               f = f + opt.L1reg * norm(parameters,1)
               gradParameters:add( sign(parameters):mul(opt.L1reg) )
            end
            if opt.L2reg ~= 0 then
    --           local norm, sign = torch.norm, torch.sign
    --           f = f + opt.L2reg * norm(parameters,2)^2/2
               parametersClone:copy(parameters)
               gradParameters:add( parametersClone:mul(opt.L2reg) )
            end
--           gradParameters:clamp(-opt.gradClip, opt.gradClip)
            return f,gradParameters
        end

        if optimMethod == optim.asgd then
            _,_,average = optimMethod(feval, parameters, optimState)
        else
--            a,b = model:parameters()
         --   print('a ' .. a[1][1][1]);
            optimMethod(feval, parameters, optimState)
         --   print('  ' .. a[1][1][1]);
        end
        model:get(1).weight:narrow(1,1,2):fill(0)
    end

    time = sys.clock() - time
    print("\n==> time for 1 epoch = " .. (time) .. ' seconds')
end

function test(inputDataTensor, inputTarget, state)
    local time = sys.clock()
    model_test:evaluate()
    local bs = opt.batchSizeTest
    local batches = inputDataTensor:size()[1]/bs
    local correct = 0
    for t = 1,batches,1 do
        local begin = (t - 1)*bs + 1
        local input = inputDataTensor:narrow(1, begin , bs)

        local pred = model_test:forward(input)
        local prob, pos = torch.max(pred, 2)
        for m = 1,bs do
           for k,v in ipairs(inputTarget[begin+m-1]) do
            if pos[m][1] == v then
                correct = correct + 1
                break
            end
          end 
        end     
    end
    state.bestAccuracy = state.bestAccuracy or 0
    state.bestEpoch = state.bestEpoch or 0
    local currAccuracy = correct/(inputDataTensor:size()[1])
    if currAccuracy > state.bestAccuracy then state.bestAccuracy = currAccuracy; state.bestEpoch = epoch end
    print(string.format("Epoch %s Accuracy: %s, best Accuracy: %s on epoch %s at time %s", epoch, currAccuracy, state.bestAccuracy, state.bestEpoch, sys.toc() ))
end






