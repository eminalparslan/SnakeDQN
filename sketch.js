let cellSize;

function setup() {
  // Square canvas
  createCanvas(windowHeight, windowHeight);
  frameRate(2);
  // Calculate size of each cell
  cellSize = windowHeight / env.cellCount;
}

function draw() {
  // Set background color
  background(54);
  // No borders
  noStroke();

  // Draw food
  let foodColor = env.colors.a;
  fill(...foodColor);
  square(env.ax * cellSize, env.ay * cellSize, cellSize);

  // Draw snake head
  let snakeHeadColor = env.colors.p;
  fill(...snakeHeadColor);
  square(env.px * cellSize, env.py * cellSize, cellSize);
  // Draw snake tail
  let snakeTailColor = env.colors.t;
  fill(...snakeTailColor);
  for (let i = 0; i < env.trail.length; i++) {
    square(env.trail[i].x * cellSize, env.trail[i].y * cellSize, cellSize);
  }
}
