class DQNAgent {
    discount = 0.99;
    replayMemorySize = 50000;
    minReplayMemorySize = 1000;
    minibatchSize = 64;
    updateTargetEvery = 5;

    constructor() {
        // Main model
        this.model = this.createModel();
        // Target model
        this.targetModel = this.createModel();
        this.targetModel.setWeights(this.model.getWeights());
        // Memory buffer
        this.replayMemory = [];
        // Counter for when to update target network
        this.targetUpdateCounter = 0;

    }

    createModel() {
        const model = tf.sequential();
        model.add(tf.layers.conv2d({
            inputShape: env.observationSpaceValues,
            kernelSize: 3,
            filters: 256,
            strides: 1,
            activation: 'relu'
        }));
        model.add(tf.layers.maxPooling2d({poolSize: 2}));
        model.add(tf.layers.dropout({rate: 0.2}));

        model.add(tf.layers.conv2d({
            kernelSize: 3,
            filters: 256,
            strides: 1,
            activation: 'relu'
        }));
        model.add(tf.layers.maxPooling2d({poolSize: 2}));
        model.add(tf.layers.dropout({rate: 0.2}));

        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({units: 64}));
        model.add(tf.layers.dense({
            units: env.actionSpaceSize,
            activation: 'linear'
        }));

        const optimizer = tf.train.adam({learningRate: 0.001});
        model.compile({
            optimizer: optimizer,
            loss: 'meanSquaredError',
            metrics: ['accuracy']
        });
        return model;
    }

    updateReplayMemory(stepResult) {
        // stepResult = {obs, action, reward, newObs, done}
        this.replayMemory.push(stepResult);
        while (this.replayMemory.length > this.replayMemorySize) {
            this.replayMemory.shift();
        }
    }

    train(terminalState) {
        if (this.replayMemory.length < this.minReplayMemorySize) {
            return;
        }

        // Randomly sampled minibatch from 
        let minibatch = _.sample(this.replayMemory, this.minibatchSize);

        // Get current states from minibatch
        let currentStatesBatch = minibatch.map(stepResult => stepResult.obs);
        // Get model predictions from current states (predicted Q values)
        // ***** maybe just predict? does tidy work?
        // ***** check for memory leaks at end
        let currentPredQsBatch = tf.tidy(() => {
            return this.model.predictOnBatch(tf.tensor(currentStatesBatch)).arraySync();
        });
        // Get model predictions for future states
        let futureStatesBatch = minibatch.map(stepResult => stepResult.newObs);
        let futurePredQsBatch = tf.tidy(() => {
            return this.model.predictOnBatch(tf.tensor(futureStatesBatch)).arraySync();
        });
        // Features and labels
        let X = [];
        let y = [];
        // Go through each step in minibatch
        for (let i = 0; i < this.minibatchSize; i++) {
            let {obs, action, reward, newObs, done} = minibatch[i];
            // Calculate new Q value
            let newQ;
            if (done) {
                newQ = reward;
            } else {
                let maxFutureQ = Math.max(...futurePredQsBatch[i]);
                // Bellman equation
                newQ = reward + this.discount * maxFutureQ;
            }
            // Update Q value for current state
            let currentPredQs = currentPredQsBatch[i];
            currentPredQs[action] = newQ;
            // Add to training data
            X.push(obs);
            y.push(currentPredQs);
        }
        // Fit on minibatch
        tf.tidy(() => {
            this.model.fit(tf.tensor(X), tf.tensor(y), {
                batchSize: this.minibatchSize,
                epochs: 1,
                verbose: 0,
                shuffle: false
            });
        });
        
        // Update target network counter every episode
        if (terminalState) {
            this.targetUpdateCounter++;
        }

        // Update target network
        if (this.targetUpdateCounter > this.updateTargetEvery) {
            this.targetModel.setWeights(this.model.getWeights());
            this.targetUpdateCounter = 0;
        }
    }

    getQs(obs) {
        return tf.tidy(() => {
            return this.model.predict(tf.tensor(obs).reshape([-1, ...env.observationSpaceValues])).arraySync()[0];
        });
    }
}
