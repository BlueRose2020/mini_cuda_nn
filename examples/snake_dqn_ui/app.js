/* UI scheduling is step-based; one timer owns training, evaluation or playback. */
'use strict';
const D = SnakeDQN, $ = id => document.getElementById(id);
const ctx = $('board').getContext('2d'), cc = $('chart').getContext('2d');
let trainer, mode = 'paused', timer = null, generation = 0, history = [], evaluations = [];
let bestScore = 0, bestPolicy = null, bestEvaluation = -Infinity, play = null, evaluation = null;
function log(text) {
  const lines = ($('log').textContent + text + '\n').split('\n');
  $('log').textContent = lines.slice(-100).join('\n');
  $('log').scrollTop = $('log').scrollHeight;
}
function stop() {
  generation++;
  if (timer !== null) clearTimeout(timer);
  timer = null; mode = 'paused'; evaluation = null;
  $('status').textContent = '已暂停';
}
function schedule(fn, delay = 0) {
  const token = generation;
  timer = setTimeout(() => {
    timer = null;
    if (token !== generation) return;
    try { fn(); } catch (error) { stop(); log('训练已停止：' + error.message); }
  }, delay);
}
function draw(e) {
  const c = 400 / e.n;
  ctx.clearRect(0, 0, 400, 400); ctx.strokeStyle = '#1b2636';
  for (let i = 1; i < e.n; i++) {
    ctx.beginPath(); ctx.moveTo(i*c, 0); ctx.lineTo(i*c, 400);
    ctx.moveTo(0, i*c); ctx.lineTo(400, i*c); ctx.stroke();
  }
  if (e.food) { ctx.fillStyle = '#ff5577'; ctx.fillRect(e.food.x*c+7, e.food.y*c+7, c-14, c-14); }
  e.snake.forEach((p, i) => { ctx.fillStyle = i ? '#47d18c' : '#80f0ba'; ctx.fillRect(p.x*c+3, p.y*c+3, c-6, c-6); });
}
function stats() {
  $('episode').textContent = trainer.episode;
  $('epsilon').textContent = trainer.eps.toFixed(3);
  $('best').textContent = bestScore;
  $('loss').textContent = trainer.loss.toFixed(4);
  $('qvalue').textContent = trainer.maxQ.toFixed(2);
  $('updates').textContent = trainer.updates;
}
function chart() {
  const w = $('chart').width, h = $('chart').height;
  cc.clearRect(0, 0, w, h);
  if (!history.length) return;
  const smooth = []; let sum = 0, lo = -1, hi = 2;
  history.forEach((v, i) => {
    sum += v; if (i >= 50) sum -= history[i-50];
    smooth.push(sum / Math.min(50, i+1)); lo = Math.min(lo, v); hi = Math.max(hi, v);
  });
  evaluations.forEach(v => {lo = Math.min(lo, v.reward); hi = Math.max(hi, v.reward);});
  const x = ep => 35 + ep / Math.max(1, history.length) * (w-45);
  const y = v => 10 + (hi-v)/(hi-lo)*(h-30);
  cc.fillStyle = '#9caac0'; cc.font = '12px sans-serif';
  cc.fillText(hi.toFixed(1), 0, 14); cc.fillText(lo.toFixed(1), 0, h-12);
  cc.strokeStyle = '#324056'; cc.beginPath(); cc.moveTo(35,y(0)); cc.lineTo(w,y(0)); cc.stroke();
  function line(points, color, width) {
    cc.strokeStyle = color; cc.lineWidth = width; cc.beginPath();
    points.forEach((p,i) => i ? cc.lineTo(x(p.episode),y(p.reward)) : cc.moveTo(x(p.episode),y(p.reward)));
    cc.stroke(); cc.lineWidth = 1;
  }
  line(history.map((v,i)=>({episode:i+1,reward:v})), '#4b689077', 1);
  line(smooth.map((v,i)=>({episode:i+1,reward:v})), '#64a4ff', 2);
  line(evaluations, '#f9c66d', 2);
}
function trainingFrame() {
  if (mode !== 'training') return;
  // Bounded work per task keeps buttons responsive even in long episodes.
  const deadline = performance.now() + 12, budget = Number($('speed').value) * 4;
  for (let i = 0; i < budget && performance.now() < deadline; i++) {
    const r = trainer.tick();
    if (!r) continue;
    history.push(r.reward); bestScore = Math.max(bestScore, r.score);
    $('reward').textContent = r.reward.toFixed(2);
    if (r.episode % 10 === 0) {
      const last = history.slice(-50), avg = last.reduce((a,b)=>a+b,0)/last.length;
      log(`ep=${r.episode} reward=${r.reward.toFixed(2)} mean50=${avg.toFixed(2)} score=${r.score} loss=${trainer.loss.toFixed(4)} |Q|=${trainer.maxQ.toFixed(2)}`);
    }
    chart();
    if (r.episode % 100 === 0) { beginEvaluation(); return; }
  }
  draw(trainer.env); stats(); schedule(trainingFrame);
}
function beginEvaluation() {
  mode = 'evaluation';
  evaluation = { policy: trainer.online.slice(), index: 0, score: 0, reward: 0, env: new D.Snake(D.random(701)) };
  $('status').textContent = '独立评估中（不更新参数）'; schedule(evaluationFrame);
}
function evaluationFrame() {
  if (mode !== 'evaluation') return;
  const e = evaluation, deadline = performance.now()+12;
  while (performance.now() < deadline) {
    const t = e.env.step(D.argmax(D.forward(e.env.state(), e.policy).q)); e.reward += t.r;
    if (e.env.done) {
      e.score += e.env.score; e.index++;
      if (e.index === 10) {
        const result = {episode: trainer.episode, score: e.score/10, reward: e.reward/10};
        evaluations.push(result); $('evaluation').textContent = result.score.toFixed(1);
        if (result.score > bestEvaluation) {bestEvaluation = result.score; bestPolicy = e.policy.slice(); $('playBest').disabled = false;}
        log(`eval ep=${trainer.episode} mean_score=${result.score.toFixed(2)} mean_reward=${result.reward.toFixed(2)} best_eval=${bestEvaluation.toFixed(2)}`);
        chart(); stats(); evaluation = null; mode = 'training'; $('status').textContent = '训练中'; schedule(trainingFrame); return;
      }
      e.env = new D.Snake(D.random(701 + e.index));
    }
  }
  schedule(evaluationFrame);
}
function playPolicy(policy, label) {
  stop(); mode = 'playing';
  // Snapshot ensures playback is the selected policy; epsilon is never changed.
  play = { policy: policy.slice(), env: new D.Snake(D.random(9001)), reward: 0, label };
  $('status').textContent = label + ' · 分数 0'; draw(play.env); schedule(playFrame, 100);
}
function playFrame() {
  if (mode !== 'playing') return;
  const t = play.env.step(D.argmax(D.forward(play.env.state(),play.policy).q)); play.reward += t.r;
  draw(play.env);
  $('status').textContent = `${play.label} · 步数 ${play.env.steps} · 分数 ${play.env.score}`;
  if (t.done) {
    mode = 'paused'; $('status').textContent += ' · 已结束（' + play.env.reason + '）';
    log(`play score=${play.env.score} reward=${play.reward.toFixed(2)} steps=${play.env.steps}`); return;
  }
  schedule(playFrame, 100);
}
function reset() {
  stop(); trainer = new D.Trainer(42); history = []; evaluations = []; bestScore = 0;
  bestPolicy = null; bestEvaluation = -Infinity; play = null;
  $('reward').textContent = '0'; $('evaluation').textContent = '—'; $('playBest').disabled = true;
  $('log').textContent = ''; stats(); chart(); draw(trainer.env); log('ready · seed=42 · Double DQN 11→32→3');
}
$('start').onclick = () => {stop(); mode='training'; $('status').textContent='训练中'; schedule(trainingFrame);};
$('pause').onclick = stop;
$('play').onclick = () => playPolicy(trainer.online, '当前策略');
$('playBest').onclick = () => {if(bestPolicy) playPolicy(bestPolicy, '最佳评估快照');};
$('reset').onclick = reset;
reset();
