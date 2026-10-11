%% comparing hamming, convolutional, and ldpc codes
% implements hamming (7,4) encode -> bsc -> decode 
% implements rate-1/2 conv. code encode -> bsc -> decode
% implements LDPC hard decision message passing decode 
%% 
message = [1 0 0 1];
probability = 0.1;
numTrials = 1000;

probabilitySweep = 0.1:0.01:.5;

figure;
semilogy(probabilitySweep, sweepResults(1, :), '-o');
hold on;
semilogy(probabilitySweep, sweepResults(2, :), '-s');
semilogy(probabilitySweep, sweepResults(3, :), '-^');
grid on;

xlabel('BSC flip probability');
ylabel('Block error rate');
legend('Hamming (7,4)', 'Convolutional', 'LDPC (8,4)', ...
       'Location', 'best');
title('Block error rate over a BSC');

function sweepResults = runSweep(numTrials, message, probabilitySweep)
    sweepResults = zeros(3, length(probabilitySweep));
    for index = 1:length(probabilitySweep)
        sweepResults(:, index) = runTrials(numTrials, message, probabilitySweep(index))';
    end 
end

function result = runTrials(numTrials, message, probability)
    countBlockErrorHamming = 0;
    countBlockErrorConvolutional = 0;
    countBlockErrorLDPC = 0;
    
    for iteration = 1:numTrials
        codewordHamming = hammingEncoder(message);
        receivedHammingCodeword = bsc(codewordHamming, probability);
        messageHamming = hammingDecode(receivedHammingCodeword);
        
        codewordConvolutional = convolutionalEncoder(message);
        receivedConvolutionalCodeword = bsc(codewordConvolutional, probability);
        messageConvolutional = convolutionalDecoder(receivedConvolutionalCodeword);
        
        codewordLDPC = encodeLDPC(message);
        receivedCodewordLDPC = bsc(codewordLDPC, probability);
        messageLDPC = decodeLDPC(receivedCodewordLDPC);
    
        countBlockErrorHamming = countBlockErrorHamming + double(~isequal(messageHamming, message));
        countBlockErrorConvolutional = countBlockErrorConvolutional + double(~isequal(messageConvolutional, message));
        countBlockErrorLDPC = countBlockErrorLDPC + double(~isequal(messageLDPC, message));
    end
    result = [countBlockErrorHamming, countBlockErrorConvolutional, countBlockErrorLDPC] ./ numTrials;
end 

%% hamming (7,4)
function decodedMessage = hammingDecode(codeword)
    errorPosition = [1 2 4]*hammingParityCheck(codeword)';
    correctedCodeword = codeword;
    if errorPosition > 0
        mask = zeros(1, 7); 
        mask(errorPosition) = 1; 
        correctedCodeword = mod(codeword + mask, 2);
    end 
    decodedMessage = correctedCodeword(logical([0 0 1 0 1 1 1]));
end

function hammingSyndrome = hammingParityCheck(codeword)
    H = [1 0 1 0 1 0 1;
         0 1 1 0 0 1 1;
         0 0 0 1 1 1 1];
    hammingSyndrome = syndrome(H, codeword);
end

function encodedMessage = hammingEncoder(data)
    encodedMessage = zeros(1, 7);
    infoMask = logical([0 0 1 0 1 1 1]);
    encodedMessage(infoMask) = data;
    parity1 = mod(encodedMessage(1) + encodedMessage(3) + encodedMessage(5) + encodedMessage(7), 2);
    parity2 = mod(encodedMessage(2) + encodedMessage(3) + encodedMessage(6) + encodedMessage(7), 2); 
    parity4 = mod(encodedMessage(4) + encodedMessage(5) + encodedMessage(6) + encodedMessage(7), 2);
    parityMask = logical([1 1 0 1 0 0 0]);
    encodedMessage(parityMask) = [parity1 parity2 parity4];
end 

%% rate 1/2, k = 3 convolutional code
function decodedMessage = convolutionalDecoder(receivedCodeword)
    timeSteps = length(receivedCodeword)/2;
    pathMetrics = inf(1,4);
    pathMetrics(1) = 0;
    priorState = zeros(timeSteps, 4);
    survivorInput = zeros(timeSteps, 4);

    for index = 1:timeSteps
        receivedPair = receivedCodeword(index*2 - 1:index*2);
        nextMetrics = inf(1,4);
        for state = 0:length(nextMetrics)-1
            memory1 = floor(state/2);
            memory2 = mod(state, 2); 
            for inputBit = [0, 1]
                output1 = mod(inputBit + memory1 + memory2, 2);
                output2 = mod(inputBit + memory2, 2); 
                nextState = inputBit*2 + memory1;
                nextStateIndex = nextState + 1;
                expectedPair = [output1 output2];
                branchMetric = sum(receivedPair ~= expectedPair);
                candidateMetric = pathMetrics(state + 1) + branchMetric;
                if candidateMetric < nextMetrics(nextStateIndex)
                    nextMetrics(nextStateIndex) = candidateMetric;
                    priorState(index, nextStateIndex) = state + 1;
                    survivorInput(index, nextStateIndex) = inputBit;
                end 
            end
        end
        pathMetrics = nextMetrics;
    end 
    decodedCodeword = zeros(1, timeSteps);
    currentStateIndex = 1;
    for index = timeSteps:-1:1
        decodedCodeword(index) = survivorInput(index, currentStateIndex);
        currentStateIndex = priorState(index, currentStateIndex);
    end 
    decodedMessage = decodedCodeword(1:end-2);
end

function codeword = convolutionalEncoder(data)
    sequence = [data, zeros(1, 2)];
    codeword = zeros(1, length(sequence)*2); 
    branch = zeros(1, 2);
    memory1 = 0;
    memory2 = 0;
    for index = 1:length(sequence)
        branch(1) = mod(sequence(index) + memory1 + memory2, 2);
        branch(2) = mod(sequence(index) + memory2, 2); 
        codeword((index)*2-1:(index)*2) = branch; 
        memory2 = memory1;
        memory1 = sequence(index);
    end
end 

%% LDPC for comparison
function message = decodeLDPC(codeword)
    H = [1 1 0 0 1 0 0 1;
         1 0 1 1 0 0 1 0;
         0 1 0 1 0 1 0 0;
         0 0 1 0 0 1 0 1;
         0 0 0 0 1 0 1 0];

    update = codeword;

    for index = 1:10
        pass = zeros(1, 5); 
        if (syndrome(H, update) == pass)
            break; 
        end 
        update = hardDecisionMessagePassing(update, H);
    end

    message = update(1:4);
end 

function codeword = encodeLDPC(data)
    H = [1 1 0 0 1 0 0 1;
         1 0 1 1 0 0 1 0;
         0 1 0 1 0 1 0 0;
         0 0 1 0 0 1 0 1;
         0 0 0 0 1 0 1 0];
    codeword = zeros(1, 8);
    codeword(1:4) = data;
    parity5 = mod(codeword(1) + codeword(3) + codeword(4), 2);
    parity6 = mod(codeword(2) + codeword(4), 2);
    parity7 = mod(codeword(1) + codeword(3) + codeword(4), 2);
    parity8 = mod(codeword(2) + codeword(3) + codeword(4), 2);
    codeword(5:8) = [parity5, parity6, parity7, parity8];
end 

%% LDPC hard decision message passing **complete for ee581
H = [0 1 0 1 1 0 0 1;
     1 1 1 0 0 1 0 0;
     0 0 1 0 0 1 1 1;
     1 0 0 1 1 0 1 0];

tx_cw = [1 0 0 1 0 1 0 1];
rx_cw = [0 1 0 1 0 1 0 1];

update = rx_cw;

for index = 1:5
    pass = zeros(1, 4); 
    if (syndrome(H, update) == pass)
        break; 
    end 
    update = hardDecisionMessagePassing(update, H);
end

LDPCDecodedCodeword = update;
%% Hard Decision Message Passing Code Helpers 
function decodedCodeword = hardDecisionMessagePassing(rx_cw, H)
    [numRows, numCols] = size(H);
    count = zeros(1, numCols);
    count(1, :) = rx_cw;
    for row = 1:numRows
        checkVals = zeros(1, length(rx_cw));
        mask = logical(H(row, :));
        checkRow = checkNodeMessages(H(row, :), rx_cw);
        checkVals(mask) = checkRow;
        count = count + checkVals;
    end 
    decodedCodeword = majorityRule(count, H);
end

function result = checkNodeMessages(node, codeword)
    mask = logical(node);
    connectedBits = codeword(mask);
    result = hardDecisionCheckMessages(connectedBits);
end 

function suggestion = hardDecisionCheckMessages(connectedBits)
    suggestion = zeros(1, length(connectedBits));
    for index = 1:length(connectedBits)
        removeMessage = connectedBits;
        removeMessage(index) = [];
        total = mod(sum(removeMessage), 2);
        suggestion(index) = total;
    end
end 

function result = syndrome(H, cw)
    result = mod(H * cw',2)';
end

function result = majorityRule(numOnes, H)
    v = ones(1, length(numOnes)); 
    totalVotes = sum(H, 1) + v;
    numZeros = totalVotes - numOnes;
    majorityCols = zeros(1, length(numOnes)); 
    majorityCols(numOnes > numZeros) = 1; 
    result = majorityCols;
end 

function receivedCodeword = bsc(transmittedCodeword, probability) 
    if ~isscalar(probability) || probability <= 0 || probability >= 1
        error('Required: 0 < p < 1');
    end 
    errorMask = rand(1, length(transmittedCodeword)) < probability;
    receivedCodeword = double(xor(logical(transmittedCodeword), errorMask)); 
end


