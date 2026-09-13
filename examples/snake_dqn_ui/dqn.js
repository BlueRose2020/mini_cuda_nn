/* Shared by the browser UI and Node regression/long-run tests. No dependencies. */
(function (root) {
  'use strict';
  const INPUTS = 11, HIDDEN = 32, ACTIONS = 3;
  const W2 = INPUTS * HIDDEN, B1 = W2 + HIDDEN * ACTIONS, B2 = B1 + HIDDEN;
  const SIZE = B2 + ACTIONS;
  function random(seed) {
    let value = seed >>> 0;
    return () => {
      value = (value + 0x6D2B79F5) >>> 0;
      let t = Math.imul(value ^ (value >>> 15), 1 | value);
      t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function network(rng) {
    const p = new Float64Array(SIZE);
    for (let i = 0; i < W2; i++) p[i] = (rng() * 2 - 1) * Math.sqrt(6 / INPUTS);
    for (let i = W2; i < B1; i++) p[i] = (rng() * 2 - 1) * Math.sqrt(6 / HIDDEN);
    return p;
  }
  function forward(s, p) {
    const h = new Float64Array(HIDDEN), q = new Float64Array(ACTIONS);
    for (let j = 0; j < HIDDEN; j++) {
      let v = p[B1 + j];
      for (let i = 0; i < INPUTS; i++) v += s[i] * p[i * HIDDEN + j];
      h[j] = Math.max(0, v);
    }
    for (let a = 0; a < ACTIONS; a++) {
      let v = p[B2 + a];
      for (let j = 0; j < HIDDEN; j++) v += h[j] * p[W2 + j * ACTIONS + a];
      if (!Number.isFinite(v)) throw new Error('Non-finite Q value; training stopped.');
      q[a] = v;
    }
    return { h, q };
  }
  function argmax(q) { let a = 0; for (let i = 1; i < q.length; i++) if (q[i] > q[a]) a = i; return a; }
  function bellman(t, online, target, gamma) {
    if (t.done) return t.r;
    const a = argmax(forward(t.ns, online).q); // Double DQN: selection != evaluation
    return t.r + gamma * forward(t.ns, target).q[a];
  }
  // The network is immutable throughout a batch. Only Q(s, chosen action) has a loss.
  function gradients(p, batch) {
    const g = new Float64Array(SIZE);
    let loss = 0, maxQ = 0;
    for (const t of batch) {
      const { h, q } = forward(t.s, p), error = q[t.a] - t.y;
      if (!Number.isFinite(t.y)) throw new Error('Non-finite TD target.');
      const d = Math.max(-1, Math.min(1, error)) / batch.length;
      loss += Math.abs(error) <= 1 ? 0.5 * error * error : Math.abs(error) - 0.5;
      for (const v of q) maxQ = Math.max(maxQ, Math.abs(v));
      g[B2 + t.a] += d;
      for (let j = 0; j < HIDDEN; j++) {
        g[W2 + j * ACTIONS + t.a] += d * h[j];
        const dh = h[j] > 0 ? d * p[W2 + j * ACTIONS + t.a] : 0;
        g[B1 + j] += dh;
        for (let i = 0; i < INPUTS; i++) g[i * HIDDEN + j] += dh * t.s[i];
      }
    }
    return { g, loss: loss / batch.length, maxQ };
  }
  class Snake {
    constructor(rng = Math.random, n = 10, maxSteps = 500) {
      this.rng = rng; this.n = n; this.maxSteps = maxSteps;
      this.snake = [{ x: Math.floor(n / 2), y: Math.floor(n / 2) }];
      this.dir = 1; this.score = 0; this.steps = 0; this.done = false; this.reason = '';
      this.spawn();
    }
    spawn() {
      const occupied = new Set(this.snake.map(p => p.y * this.n + p.x)), free = [];
      for (let i = 0; i < this.n * this.n; i++) if (!occupied.has(i)) free.push(i);
      if (!free.length) { this.food = null; this.done = true; this.reason = 'win'; return; }
      const cell = free[Math.floor(this.rng() * free.length)];
      this.food = { x: cell % this.n, y: Math.floor(cell / this.n) };
    }
    next(dir) {
      const h = this.snake[0];
      return { x: h.x + (dir === 1) - (dir === 3), y: h.y + (dir === 2) - (dir === 0) };
    }
    collision(p) {
      if (p.x < 0 || p.x >= this.n || p.y < 0 || p.y >= this.n) return true;
      const eats = this.food && p.x === this.food.x && p.y === this.food.y;
      // The tail vacates its cell on a non-growing move.
      return this.snake.slice(0, this.snake.length - (eats ? 0 : 1)).some(b => b.x === p.x && b.y === p.y);
    }
    state() {
      const h = this.snake[0], food = this.food || h;
      return [+this.collision(this.next(this.dir)), +this.collision(this.next((this.dir + 3) % 4)),
        +this.collision(this.next((this.dir + 1) % 4)), +(this.dir === 0), +(this.dir === 1),
        +(this.dir === 2), +(this.dir === 3), +(food.y < h.y), +(food.x > h.x),
        +(food.y > h.y), +(food.x < h.x)];
    }
    step(a) {
      if (this.done) throw new Error('Episode has ended.');
      if (![0, 1, 2].includes(a)) throw new Error('Invalid action.');
      this.dir = (this.dir + (a === 0 ? 3 : a === 2 ? 1 : 0)) % 4;
      this.steps++;
      const p = this.next(this.dir);
      let r = -0.01;
      if (this.collision(p)) { this.done = true; this.reason = 'collision'; r = -1; }
      else {
        this.snake.unshift(p);
        if (p.x === this.food.x && p.y === this.food.y) { r = 1; this.score++; this.spawn(); }
        else this.snake.pop();
      }
      // The UI game ends at 500 moves: this transition must not bootstrap.
      if (!this.done && this.steps >= this.maxSteps) { this.done = true; this.reason = 'limit'; }
      return { ns: this.state(), r, done: this.done };
    }
  }
  class Trainer {
    constructor(seed = 42) {
      this.rng = random(seed); this.envRng = random(seed ^ 0xABC123);
      this.online = network(this.rng); this.target = this.online.slice();
      this.m = new Float64Array(SIZE); this.v = new Float64Array(SIZE);
      this.replay = []; this.replayIndex = 0; this.batchSize = 32; this.capacity = 20000;
      this.gamma = 0.95; this.lr = 0.0005; this.tau = 0.005; this.clip = 5;
      this.steps = 0; this.updates = 0; this.episode = 0; this.eps = 1;
      this.loss = 0; this.maxQ = 0; this.gradNorm = 0;
      this.env = new Snake(this.envRng); this.total = 0;
    }
    update() {
      const batch = [];
      for (let i = 0; i < this.batchSize; i++) {
        const t = this.replay[Math.floor(this.rng() * this.replay.length)];
        batch.push({ s: t.s, a: t.a, y: bellman(t, this.online, this.target, this.gamma) });
      }
      const { g, loss, maxQ } = gradients(this.online, batch);
      let norm = 0;
      for (const x of g) norm += x * x;
      norm = Math.sqrt(norm);
      if (![loss, maxQ, norm].every(Number.isFinite)) throw new Error('Non-finite batch; training stopped.');
      this.loss = loss; this.maxQ = maxQ; this.gradNorm = norm;
      const factor = Math.min(1, this.clip / (norm || 1));
      const step = this.updates + 1, c1 = 1 - Math.pow(0.9, step), c2 = 1 - Math.pow(0.999, step);
      // Validate all candidate updates before committing any parameter.
      const next = this.online.slice(), m = this.m.slice(), v = this.v.slice();
      for (let i = 0; i < SIZE; i++) {
        const d = g[i] * factor;
        m[i] = 0.9 * m[i] + 0.1 * d; v[i] = 0.999 * v[i] + 0.001 * d * d;
        next[i] -= this.lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8);
        if (!Number.isFinite(next[i])) throw new Error('Non-finite update; training stopped.');
      }
      this.online = next; this.m = m; this.v = v; this.updates = step;
      for (let i = 0; i < SIZE; i++) this.target[i] += this.tau * (next[i] - this.target[i]);
    }
    tick() {
      const s = this.env.state();
      const a = this.rng() < this.eps ? Math.floor(this.rng() * ACTIONS) : argmax(forward(s, this.online).q);
      const t = this.env.step(a);
      const transition = { s, a, ...t };
      if (this.replay.length < this.capacity) this.replay.push(transition);
      else { this.replay[this.replayIndex] = transition; this.replayIndex = (this.replayIndex + 1) % this.capacity; }
      this.total += t.r; this.steps++;
      if (this.replay.length >= 256 && this.steps % 4 === 0) this.update();
      if (!t.done) return null;
      const result = { episode: ++this.episode, reward: this.total, score: this.env.score,
        steps: this.env.steps, reason: this.env.reason };
      this.eps = Math.max(0.1, Math.pow(0.995, this.episode));
      this.env = new Snake(this.envRng); this.total = 0;
      return result;
    }
  }
  // Evaluation/playback use a snapshot + their own RNG, never the trainer's epsilon or RNG.
  function evaluate(p, seeds = [701, 702, 703, 704, 705, 706, 707, 708, 709, 710]) {
    let reward = 0, score = 0;
    for (const seed of seeds) {
      const e = new Snake(random(seed));
      while (!e.done) { const t = e.step(argmax(forward(e.state(), p).q)); reward += t.r; }
      score += e.score;
    }
    return { reward: reward / seeds.length, score: score / seeds.length };
  }
  const api = { random, network, forward, argmax, gradients, bellman, Snake, Trainer, evaluate, SIZE, W2, B1, B2 };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SnakeDQN = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
