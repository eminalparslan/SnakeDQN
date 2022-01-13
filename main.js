const env = new SnakeEnv();
const agent = new DQNAgent();

// ****** print some stuff
// ****** tensorboard
// ****** score (num food eaten display on canvas)

// Environment settings
const EPISODES = 20000;
// Exploration settings
let epsilon = 1;
const EPSILON_DECAY = 0.99975;
const MIN_EPSILON = 0.001;
// Stats settings
const ROLLING_AVERAGE_RANGE = 50;

let epRewards = [];
let avgRewards = [];

// const metrics = ['loss', 'acc'];
// const container = {
//     name: 'Model', tab: 'Training', styles: {height: '1000px'}
// }

for (let episode = 1; episode <= EPISODES; episode++) {
    // Restart episode
    let episodeReward = 0;
    let step = 1;
    let done = false;

    // Reset environment and get initial state
    let obs = env.reset();

    // Finish episode
    while (!done) {
        let action, reward, newObs;
        // Choose action
        if (Math.random() > epsilon) {
            // Argmax on predicted q values gives us action
            action = agent.getQs(obs).reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
        } else {
            // Randint for random action
            action = Math.floor(Math.random() * env.actionSpaceSize);
        }
        // Take a step
        ({newObs, reward, done} = env.step(action));
        // Update total episode reward
        episodeReward += reward;
        // Add to memory
        agent.updateReplayMemory({obs, action, reward, newObs, done});
        // Train the model
        agent.train(done);
        // Update state
        obs = newObs;
        // Update step counter
        step++;
    }
    // Keep track of episode rewards
    epRewards.push(episodeReward);
    // Calculate rolling average
    let rewardWindow = epRewards.slice(-Math.min(ROLLING_AVERAGE_RANGE, epRewards.length));
    let avgReward = rewardWindow.reduce((a, b) => a + b) / rewardWindow.length;
    avgRewards.push(avgReward);

    // Decay epsilon
    if (epsilon > MIN_EPSILON) {
        epsilon *= EPSILON_DECAY;
        epsilon = Math.max(MIN_EPSILON, epsilon);
    }

    console.log("Episode: " + episode);
    console.log("Steps taken: " + step);
    console.log("Average reward: " + avgReward);
}
