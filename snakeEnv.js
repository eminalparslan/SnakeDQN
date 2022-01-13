class SnakeEnv {
    // Cells per row/col
    cellCount = 20;
    // Penalty for each move to encourage more efficient paths to food
    movePenalty = 1;
    // Huge penalty for death
    deathPenalty = 100;
    // Decent reward for eating food (only way to get rewarded)
    foodReward = 25;
    // 3 for RGB values
    observationSpaceValues = [this.cellCount, this.cellCount, 3];
    actionSpaceSize = 4;
    // Colors for each type of cell (p: snake head, t: snake tail, a: food)
    colors = {
        p: [0, 0, 255],
        t: [0, 255, 0],
        a: [255, 0, 0]
    }

    getImage() {
        // Zeroed out image
        let img = tf.tidy(() => {
            return tf.zeros(this.observationSpaceValues).arraySync();
        });
        // Populate image
        img[this.ax][this.ay] = this.colors.a.map(x => x / 255.0);
        img[this.px][this.py] = this.colors.p.map(x => x / 255.0);
        for (let i = 0; i < this.trail.length; i++) {
            img[this.trail[i].x][this.trail[i].y] = this.colors.t.map(x => x / 255.0);
        }
        return img;
    }

    reset() {
        this.px = this.py = 10;
        this.vx = this.vy = 0;
        this.ax = Math.floor(Math.random() * this.cellCount);
        this.ay = Math.floor(Math.random() * this.cellCount);
        this.tail = 5;
        this.trail = [];
        this.episodeStep = 0;
        return this.getImage();
    }

    step(action) {
        // Keep track of steps in episode
        this.episodeStep++;
        // Set velocity based on action
        switch (action) {
            // left
            case 0:
                this.vx = -1, this.vy = 0;
                break;
            // down
            case 1:
                this.vx = 0, this.vy = 1;
                break;
            // right
            case 2:
                this.vx = 1, this.vy = 0;
                break;
            //up
            case 3:
                this.vx = 0, this.vy = -1;
                break;
        }
        // Add head to tail and limit length
        this.trail.push({x: this.px, y: this.py});
        while (this.trail.length > this.trail) {
            this.trail.shift();
        }
        // Move based on action
        this.px += this.vx;
        this.py += this.vy;
        // Done iff dead
        let done = false;
        // Penalty for every move
        let reward =  0 - this.movePenalty;
        // Check for wall collision
        if (this.px < 0 || this.px > this.cellCount - 1 ||
            this.py < 0 || this.py > this.cellCount - 1) {
            reward -= this.deathPenalty;
            done = true;
        }
        // Check for self collision
        for (let i = 0; i < this.trail.length; i++) {
            if (this.trail[i].x == this.px && this.trail[i].y == this.py) {
                reward -= this.deathPenalty;
                done = true;
            }
        }
        // Check if food eaten
        if (this.ax == this.px && this.ay == this.py) {
            this.tail++;
            this.ax = Math.floor(Math.random() * this.cellCount);
            this.ay = Math.floor(Math.random() * this.cellCount);
            reward += this.foodReward;
            // Make sure food doesn't spawn on snake
            // If it does, search linearly until open spot found
            while (this.trail.some(e => e.x == this.ax && e.y == this.ay)) {
                this.ax++;
                // Wrap around
                if (this.ax > this.cellCount - 1) {
                    this.ax = 0;
                    this.ay++;
                }
                if (this.ay > this.cellCount - 1) {
                    this.ay = 0;
                }
            }
        }
        // Make observation
        let newObs = this.getImage();
        return {newObs, reward, done};
    }
}
